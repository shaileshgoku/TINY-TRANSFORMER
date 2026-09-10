
import streamlit as st
import torch
import json

from model import TinyTransformer


# ============================================================
# Configuration
# ============================================================

CONTEXT_LENGTH = 4
EMBED_DIM = 4
NUM_HEADS = 2
FF_DIM = 8
NUM_BLOCKS = 2


# ============================================================
# Load vocabulary
# ============================================================

with open("vocab.json", "r") as f:
    vocab = json.load(f)

token_to_id = vocab["token_to_id"]

id_to_token = {
    int(k): v
    for k, v in vocab["id_to_token"].items()
}

vocab_size = len(token_to_id)


# ============================================================
# Load trained model
# ============================================================

@st.cache_resource
def load_model():

    model = TinyTransformer(
        vocab_size=vocab_size,
        context_length=CONTEXT_LENGTH,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_blocks=NUM_BLOCKS
    )

    model.load_state_dict(
        torch.load(
            "model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model()


# ============================================================
# Text generation
# ============================================================

def generate_text(
    start_text,
    max_new_tokens=10
):

    tokens = start_text.split()

    # Check whether every word exists
    # in our tiny vocabulary
    unknown_words = [
        token
        for token in tokens
        if token not in token_to_id
    ]

    if unknown_words:

        return (
            "I don't know these words yet: "
            + ", ".join(unknown_words)
        )

    generated_ids = [
        token_to_id[token]
        for token in tokens
    ]

    for _ in range(max_new_tokens):

        # Keep only the most recent context
        input_ids = generated_ids[-CONTEXT_LENGTH:]

        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long
        )

        with torch.no_grad():

            logits = model(input_tensor)

        # Look at prediction for the final token
        next_token_logits = logits[
            0,
            -1
        ]

        # Choose highest-probability token
        next_token_id = torch.argmax(
            next_token_logits
        ).item()

        next_token = id_to_token[
            next_token_id
        ]

        # Stop internally at <END>
        if next_token == "<END>":
            break

        generated_ids.append(
            next_token_id
        )

    generated_tokens = [
        id_to_token[token_id]
        for token_id in generated_ids
    ]

    return " ".join(generated_tokens)


# ============================================================
# Streamlit Interface
# ============================================================

st.set_page_config(
    page_title="Tiny Transformer",
    page_icon="🤖"
)

st.title("🤖 Tiny Transformer")

st.write(
    "A tiny language model built from scratch "
    "using PyTorch."
)

st.caption(
    "Transformer architecture • Multi-Head Attention • "
    "Feed-Forward Network • Autoregressive Generation"
)

st.divider()


# User input
user_input = st.text_input(
    "Enter a starting sentence:",
    placeholder="Try: I love"
)


# Generation settings
max_tokens = st.slider(
    "Maximum new tokens",
    min_value=1,
    max_value=10,
    value=5
)


# Generate button
if st.button("Generate Text"):

    if not user_input.strip():

        st.warning(
            "Please enter some text."
        )

    else:

        result = generate_text(
            user_input.strip(),
            max_new_tokens=max_tokens
        )

        st.subheader("Generated Text")

        st.success(result)


st.divider()

st.write("### Try these examples")

st.write("""
- `I love`
- `Cats`
- `Dogs`
- `Cats are`
- `Dogs are`
""")

st.caption(
    "This is an educational Tiny Transformer trained "
    "on a very small dataset."
)
