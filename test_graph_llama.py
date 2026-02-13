import torch
from model import LlamaConfig, GraphLlamaForCausalLM

def main():
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64
    )

    graph_config = {
        "node_dim": 64,
        "num_layers": 2,
        "encoder_type": "gat"
    }

    model = GraphLlamaForCausalLM(config, graph_config=graph_config)

    # 假数据
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    graph_data = {
        "node_features": torch.randn(5, 64),
        "edge_index": torch.tensor([[0,1,1,2],[1,0,2,1]], dtype=torch.long)
    }

    logits = model(input_ids, graph_data=graph_data)
    print("logits shape:", logits.shape)

if __name__ == "__main__":
    main()