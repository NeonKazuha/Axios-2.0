import streamlit as st
import torch
import torch.nn.functional as F

from axiosthingy import BigramLanguageModel, device, block_size, encode, decode

@st.cache_resource
def load_model():
    model = BigramLanguageModel()
    model.load_state_dict(torch.load('Shakespeare_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

st.title('Shakespearean Text Generator')

input_text = st.text_area('Enter some starting text:', 'In times long past')
max_new_tokens = st.slider('Number of tokens to generate:', 1, 10000, 100)

if st.button('Generate Text'):
    context = torch.tensor(encode(input_text), dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)
    
    generated_text = decode(generated_tokens[0].tolist())
    
    st.write('Generated Text:')
    st.write(generated_text)

st.sidebar.header('Model Information')
st.sidebar.write(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
st.sidebar.write(f'Embedding Dimension: {model.token_embedding_table.embedding_dim}')
st.sidebar.write(f'Number of Layers: {len(model.blocks)}')
