// ref: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MultinomialKernel.cu

namespace at::native {

// 匿名命名空间，用于限制下面函数的作用域
namespace {

// 函数模板，用于判断一个数是否为无穷大
template <
    typename T,
    typename = std::enable_if_t<
        std::is_floating_point_v<T> || std::is_convertible_v<T, float>>>
inline __device__ bool _isinf(T x) {
  // 如果 T 是浮点数类型，直接调用 ::isinf
  if constexpr (std::is_floating_point_v<T>) {
    return ::isinf(x);
  } else {
    // 如果 T 不是浮点数类型，先转换为 float 再调用 ::isinf
    return ::isinf(static_cast<float>(x));
  }
}

// 定义最大块数
#define MAX_NUM_BLOCKS 200

// 将每一行的 L1 范数归一化为 1，用于多项分布
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(cuda::detail::CUDA_NUM_THREADS)
__global__ void renormRowsL1(scalar_t* dist, long rows, long cols) {
  // 定义共享内存
  extern __shared__  unsigned char my_smem[];
  scalar_t *smem = reinterpret_cast<scalar_t *>(my_smem);
  scalar_t zero = static_cast<scalar_t>(0);
  scalar_t val;
  // 遍历每一行
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    scalar_t sum = static_cast<scalar_t>(0);
    // 遍历每一列，计算当前行的和
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
      val = dist[row * cols + col];
      CUDA_KERNEL_ASSERT(!(val < zero)); // 处理 NaN
      sum = sum + val;
    }

    // 归约求和
    sum = cuda_utils::BlockReduceSum(sum, smem);
    if (threadIdx.x == 0) {
      CUDA_KERNEL_ASSERT(!(val < zero)); // 处理 NaN
      smem[0] = sum;
    }
    __syncthreads();

    sum = smem[0];
    // 如果和大于 0，则归一化当前行
    if (sum > zero) {
      for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
        dist[row * cols + col] = dist[row * cols + col] / sum;
      }
    }
  }
}

// 归一化每一行
void renormRows(Tensor& t) {
  // 检查 t 的维度是否为 2
  TORCH_CHECK(t.dim() == 2);
  int64_t rows = t.size(0);
  int64_t cols = t.size(1);

  // 获取当前设备属性
  auto props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props != nullptr);
  int numSM = props->multiProcessorCount;
  const int64_t maxThreads = std::min(
      props->maxThreadsPerBlock, cuda_utils::kCUDABlockReduceMaxThreads);

  int warp_size = at::cuda::warp_size();
  dim3 grid(rows < numSM * 4 ? rows : numSM * 4);
  dim3 block(std::min(maxThreads, warp_size * ceil_div(cols, int64_t{warp_size})));

  // 调用 renormRowsL1 内核函数
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, t.scalar_type(), "renormRows_cuda", [&] {
    renormRowsL1<scalar_t>
        <<<grid, block, (block.x / warp_size) * sizeof(scalar_t),
        at::cuda::getCurrentCUDAStream()>>>(t.mutable_data_ptr<scalar_t>(),
            rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

// 二分查找函数，用于多项分布
template <typename scalar_t>
__device__ int binarySearchForMultinomial(const scalar_t* cumdist,
                                          const scalar_t* dist,
                                          int size,
                                          scalar_t val) {
  int start = 0;
  int end = size;
  // 确保 cumdist[size - 1] > 0
  CUDA_KERNEL_ASSERT(cumdist[size - 1] > static_cast<scalar_t>(0));

  // 二分查找
  while (end - start > 0) {
    int mid = start + (end - start) / 2;

    scalar_t midVal = cumdist[mid];
    if (midVal < val) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }

  // 处理特殊情况
  if (start == size) {
    start = size - 1;
  }

  // 找到第一个非零元素
  while(start >= 1 && dist[start] == 0) start--;

  return start;
}

// 核函数，用于有放回的多项分布采样
template <typename scalar_t>
__global__ void
sampleMultinomialWithReplacement(PhiloxCudaState philox_args,
                                 int totalSamples,
                                 int64_t* dest,
                                 int64_t distributions,
                                 int categories,
                                 const scalar_t* normDistPrefixSum,
                                 const scalar_t* normDist) {
  // 当前种子的解包
  auto seeds = at::cuda::philox::unpack(philox_args);

  // 计算全局索引
  int idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

  // 初始化随机数状态
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds),
              idx,
              std::get<1>(seeds),
              &state);

  // 每个块处理一个分布
  for (int64_t curDist = blockIdx.y;
       curDist < distributions;
       curDist += gridDim.y) {
    for (int sample = blockIdx.x*blockDim.x + threadIdx.x;
         sample < totalSamples; sample += blockDim.x*gridDim.x) {

      // 生成随机数
      auto rand = curand_uniform4(&state);
      scalar_t r = static_cast<scalar_t>(rand.x);

      // 查找随机数所在的桶
      int choice = binarySearchForMultinomial<scalar_t>(
          normDistPrefixSum + curDist * categories,
          normDist + curDist * categories,
          categories,
          r);

      // 保存结果
      dest[curDist * totalSamples + sample] = choice;

    }
  }
}

// 核函数，用于一次性多项分布采样
template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(cuda::detail::CUDA_NUM_THREADS)
__global__ void sampleMultinomialOnce(
    int64_t* dest,
    int64_t distributions,
    int categories,
    const scalar_t* sampled,
    const scalar_t* dist,
    int stride_dist, // dist->stride(0)
    int stride_categories // dist->stride(1)
) {
  // 定义共享内存
  extern __shared__  unsigned char my_smem[];
  __shared__ bool found;
  __shared__ unsigned foundPos;

  accscalar_t *smem = reinterpret_cast<accscalar_t *>(my_smem);

  accscalar_t accZero = static_cast<accscalar_t>(0);
  scalar_t zero = static_cast<scalar_t>(0);

  // 每个块处理一个分布
  for (int64_t curDist = blockIdx.x;
       curDist < distributions; curDist += gridDim.x) {
    // 计算当前分布的和
    accscalar_t sum = accZero;
    scalar_t val;
    for (int cat = threadIdx.x; cat < categories; cat += blockDim.x) {
      val = dist[curDist * stride_dist + cat * stride_categories];
      CUDA_KERNEL_ASSERT(!at::_isnan(val));
      CUDA_KERNEL_ASSERT(!_isinf(val));
      CUDA_KERNEL_ASSERT(!(val < zero));
      sum = sum + static_cast<accscalar_t>(val);
    }

    // 归约求和
    sum = cuda_utils::BlockReduceSum(sum, smem);

    // 广播和和采样值
    if (threadIdx.x == 0) {
      CUDA_KERNEL_ASSERT(!_isinf(val));
      CUDA_KERNEL_ASSERT(sum > accZero);

      foundPos = 0;
      smem[0] = sum;
      smem[1] = sampled[curDist];
    }
    __syncthreads();

    sum = smem[0];
    scalar_t sample = static_cast<scalar_t>(smem[1]);
    __syncthreads();

    // 如果和为 0，选择第一个元素
    if (sum == accZero) {
      if (threadIdx.x == 0) {
        dest[curDist] = 0;
      }
      continue;
    }

    int chunks = (categories + (int)blockDim.x - 1) / blockDim.x;
    accscalar_t prevHighProb = accZero;
    found = false;

    // 分块处理
    for (int chunk = 0; chunk < chunks && !found; ++chunk) {
      int cat = chunk * blockDim.x + threadIdx.x;

      accscalar_t dist_val = cat < categories ?
                             static_cast<accscalar_t>(dist[curDist * stride_dist + cat * stride_categories]) / sum :
                             accZero;

      smem[threadIdx.x] = dist_val;
      __syncthreads();

      // 前缀和
      for (int offset = 1; offset < blockDim.x; offset *= 2) {
        accscalar_t val = accZero;

        if (threadIdx.x >= offset) {
          val = smem[threadIdx.x - offset] + smem[threadIdx.x];
        }

        __syncthreads();
        if (threadIdx.x >= offset) {
          smem[threadIdx.x] = val;
        }
        __syncthreads();
      }

      // 检查采样值是否在当前桶中
      scalar_t curBucket =
          static_cast<scalar_t>(smem[threadIdx.x] + prevHighProb);
      scalar_t prevBucket = static_cast<scalar_t>(
          threadIdx.x == 0 ? prevHighProb
                          : smem[threadIdx.x - 1] + prevHighProb);
      bool inBucket =
          (cat < categories) &&
          (!(sample >= curBucket) &&
          (sample >= prevBucket) &&
          (dist_val > zero));

      if (inBucket) {
        // 找到采样值
        atomicMax(&foundPos, cat);
        found = true;
      }

      // 存储前缀和的高值
      prevHighProb = prevHighProb + smem[blockDim.x - 1];

      __syncthreads();
    }

    // 保存结果
    if (threadIdx.x == 0) {
      if (found) {
          dest[curDist] = foundPos;
      } else {
        // 处理稀有错误情况
        for (int cat = categories - 1; cat >= 0; --cat) {
          if (dist[curDist * stride_dist + cat * stride_categories] > zero) {
            dest[curDist] = cat;
            break;
          }
        }
      }
    }
  }
}

// 实现有放回的多项分布采样
void multinomial_with_replacement_kernel_impl(
    Tensor& result,
    const Tensor& self,
    const int64_t n_sample,
    std::optional<Generator> generator) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(generator, cuda::detail::getDefaultCUDAGenerator());

  int inputSize = self.dim();
  int64_t numDist =
      inputSize == 1 ? 1 : self.size(0);
  int numCategories =
      inputSize == 1 ? self.size(0) : self.size(1);

  // 重构数据为 2D
  auto self_v = inputSize == 1 ? self.view({numDist, numCategories}) : self;

  result.resize_({numDist, n_sample});

  // 调用不同类型的多项分布采样核函数
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self_v.scalar_type(), "multinomial_kernel_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props != nullptr);
    int numSM = props->multiProcessorCount;
    int maxThreads = props->maxThreadsPerBlock;
    int maxShared = props->sharedMemPerBlock;

    int warp_size = at::cuda::warp_size();
    int requiredWarps = at::ceil_div(numCategories, warp_size);
    int requiredThreads = std::min(maxThreads, requiredWarps * warp_size);
    int requiredShared = requiredThreads * sizeof(accscalar_t);

    // 优化的无分配实现
    if (n_sample == 1 && maxShared >= requiredShared) {
      Tensor sampled = at::detail::empty_cuda({numDist, n_sample}, self_v.options());
      at::native::uniform_(sampled, 0.0, 1.0, generator);

      dim3 block(requiredThreads);
      dim3 grid(std::min(static_cast<int>(numDist), numSM * 4));

      sampleMultinomialOnce<scalar_t, accscalar_t>
          <<<grid, block,
          requiredShared,
          at::cuda::getCurrentCUDAStream()>>>(
              result.mutable_data_ptr<int64_t>(),
                  numDist,
                  numCategories,
                  sampled.const_data_ptr<scalar_t>(),
                  self_v.const_data_ptr<scalar_t>(),
                  self_v.stride(0),
                  self_v.stride(1)
          );
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      // 通用的慢速实现，带有内存分配

      Tensor origDist = native::empty_like(
          self_v,
          std::nullopt /* dtype */,
          std::nullopt /* layout */,
          std::nullopt /* device */,
          std::nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      origDist.copy_(self_v);

      Tensor normDist = native::empty_like(
          self_v,
          std::nullopt /* dtype */,
          std::nullopt /* layout */,
          std::nullopt /* device */,
          std::nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      Tensor prefixSum = native::empty_like(
          self_v,
          std::nullopt /* dtype */,
          std::nullopt /* layout */,
          std::nullopt /* device */,
          std::nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      // 归一化每一行
      normDist.copy_(origDist);
      renormRows(normDist);

      // 计算前缀和
      at::cuda::cumsum_out(prefixSum, normDist, 1);

      PhiloxCudaState rng_engine_inputs;

        // 二分查找是 warp 分支（所以实际上我们只运行一个线程），但为了更好地利用，
        // 需要每个块至少有 4 个 warp。
        dim3 block(128);

        // 每个块将并发地从一个分布中生成一个样本。
        int grid_y=std::min<int>(numDist, at::cuda::getCurrentDeviceProperties()->maxGridSize[1]);
        dim3 grid((n_sample-1)/block.x+1, grid_y);
        {
          std::lock_guard<std::mutex> lock(gen->mutex_);

          // 每个线程为 (numdist/numblocks.y) 分布生成一个样本，但是由于我们必须使用
          // curand_uniform4，偏移量是 4 倍。
          auto offset = ((numDist-1)/grid.y+1)*4;
          rng_engine_inputs = gen->philox_cuda_state(offset);
        }
        // 有放回的采样

        sampleMultinomialWithReplacement
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                rng_engine_inputs,
                n_sample,
                result.mutable_data_ptr<int64_t>(),
                numDist, numCategories,
                prefixSum.const_data_ptr<scalar_t>(),
                normDist.const_data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  });

  // 如果输入维度为 1，调整结果维度
  if (inputSize == 1) {
    result.resize_({n_sample});
  }
}
}

// 注册多项分布采样内核实现
REGISTER_DISPATCH(
    multinomial_with_replacement_stub,
    &multinomial_with_replacement_kernel_impl);
} // namespace at::native