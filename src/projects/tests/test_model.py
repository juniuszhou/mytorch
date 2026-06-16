from projects.model import TransformerLM, load_model, load_model_safe, save_model, save_model_safe


def test_save_and_load_model():
    model = TransformerLM(
        vocab_size=21128,
        context_length=128,
        d_model=32,
        num_layers=4,
        num_heads=8,
        d_ff=64,
        rope_theta=10000.0,
    )
    save_model(model)

    loaded = load_model(model, "model.pth")
    assert loaded.vocab_size == model.vocab_size
    assert type(loaded) is TransformerLM


def test_save_and_load_model_safe():
    model = TransformerLM(
        vocab_size=21128,
        context_length=128,
        d_model=32,
        num_layers=4,
        num_heads=8,
        d_ff=64,
        rope_theta=10000.0,
    )
    save_model_safe(model, name="test-checkpoint")

    loaded = load_model_safe("test-checkpoint")
    assert loaded.vocab_size == model.vocab_size
    assert loaded.d_model == model.d_model
    assert type(loaded) is TransformerLM
