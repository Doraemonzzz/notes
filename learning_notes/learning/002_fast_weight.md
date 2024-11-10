

$$
\boldsymbol{F}_\theta(\boldsymbol{W}(s), \boldsymbol{x}(s))=\sigma(\beta(s)) \begin{cases}\boldsymbol{k}(s) \otimes \boldsymbol{v}(s) & \text { Hebb-style } \\ \boldsymbol{v}(s) \otimes\left(\boldsymbol{k}(s)-\boldsymbol{W}(s)^{\top} \boldsymbol{v}(s)\right) & \text { Oja-style } \\ (\boldsymbol{v}(s)-\boldsymbol{W}(s) \boldsymbol{k}(s)) \otimes \boldsymbol{k}(s) & \text { Delta-style }\end{cases}
$$


## Delta

$$
\begin{aligned}
& \boldsymbol{q}_t, \boldsymbol{k}_t, \boldsymbol{v}_t, \beta_t=\boldsymbol{W}^{\text {slow }} \boldsymbol{x}_t \\
& \overline{\boldsymbol{v}}_t=\boldsymbol{W}_{t-1} \phi\left(\boldsymbol{k}_t\right) \\
& \boldsymbol{W}_t=\boldsymbol{W}_{t-1}+\sigma\left(\beta_t\right)\left(\boldsymbol{v}_t-\overline{\boldsymbol{v}}_t\right) \otimes \phi\left(\boldsymbol{k}_t\right)
\end{aligned}
$$

## Recurrent Delta

$$
\boldsymbol{q}_t, \boldsymbol{k}_t, \boldsymbol{v}_t, \beta_t=\boldsymbol{W}^{\text {slow }}\left[\boldsymbol{x}_t, \tanh \left(\boldsymbol{y}_{t-1}\right)\right]
$$


## self-referential weight matrix (SRWM)

$$
\begin{gathered}
\boldsymbol{y}_t, \boldsymbol{k}_t, \boldsymbol{q}_t, \beta_t=\boldsymbol{W}_{t-1} \boldsymbol{x}_t \\
\boldsymbol{v}_t=\boldsymbol{W}_{t-1} \phi\left(\boldsymbol{q}_t\right) ; \overline{\boldsymbol{v}}_t=\boldsymbol{W}_{t-1} \phi\left(\boldsymbol{k}_t\right) \\
\boldsymbol{W}_t=\boldsymbol{W}_{t-1}+\sigma\left(\beta_t\right)\left(\boldsymbol{v}_t-\overline{\boldsymbol{v}}_t\right) \otimes \phi\left(\boldsymbol{k}_t\right)
\end{gathered}
$$