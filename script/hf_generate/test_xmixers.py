import torch
import torch.nn as nn

class MinimalModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.channel_norm = nn.LayerNorm(hidden_size)
        self.channel_mixer = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, past_key_values=None):
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len, hidden_size)
            past_key_values: Tensor of shape (batch_size, past_len, hidden_size)
        Returns:
            output: Tensor of shape (batch_size, total_seq_len, hidden_size)
            new_past_key_values: Tensor of shape (batch_size, total_seq_len, hidden_size)
        """
        if past_key_values is not None:
            # Concatenate past and current input
            input_ids = torch.cat([past_key_values, input_ids], dim=1)

        # Apply channel_norm
        normed = self.channel_norm(input_ids)

        # Apply channel_mixer
        mixed = self.channel_mixer(normed)

        # Update past_key_values
        new_past_key_values = input_ids

        return mixed, new_past_key_values

# Example usage
def test_minimal_model():
    batch_size = 2
    seq_len = 4
    past_len = 3
    hidden_size = 8

    model = MinimalModel(hidden_size)
    model.eval()

    # Random input tensors
    current_input = torch.randn(batch_size, seq_len, hidden_size)
    past_input = torch.randn(batch_size, past_len, hidden_size)

    # Forward pass without past_key_values
    output, new_past_key_values = model(current_input)
    print("Without past_key_values:")
    print("Output shape:", output.shape)
    print("New past_key_values shape:", new_past_key_values.shape)

    # Forward pass with past_key_values
    output, new_past_key_values = model(current_input, past_input)
    print("With past_key_values:")
    print("Output shape:", output.shape)
    print("New past_key_values shape:", new_past_key_values.shape)

if __name__ == "__main__":
    test_minimal_model()
