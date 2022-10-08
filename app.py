import streamlit as st
import torch
import boto3
import io
import pandas as pd

from PIL import Image
from torchvision.transforms import ToTensor
import os
from datetime import datetime

MODEL_PATH = os.getenv('model', 's3://emlov2-s5/model.trace.pt')
FLAGGED_DIR = os.getenv('flagged_dir', 's3://emlov2-s5/outputs')

from typing import Dict

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
            
# @st.cache
def load_model(model):
    jit_path = os.path.join(os.getcwd(), model)
    model = torch.jit.load(jit_path)
    model.eval()
    # get the classnames
    with open("cifar10_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return model, categories

# @st.cache
def predict(inp_img: Image, top_k: int) -> Dict[str, float]:
    inp_img = ToTensor()(inp_img) # the input image is scaled to [0.0, 1.0]
    inp_img = inp_img.unsqueeze(0)
    inp_img = inp_img.to(torch.float32, copy=True)
    model, categories = load_model('model.trace.pt')
    topk_ids = []
    # inference
    out = model.forward_jit(inp_img)
    topk = out.topk(top_k)[1][0]
    topk_ids.append(topk.cpu().numpy())
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    topk_prob, topk_label = torch.topk(probabilities, top_k)
    confidences = {}
    confidences['Labels'] = [categories[topk_label[i]] for i in range(topk_prob.size(0))]
    confidences['Confidence (%)'] = [f"{round(float(topk_prob[i]),2) *100:.2f}" for i in range(topk_prob.size(0))]
    return confidences

def split_s3_path(s3_path):
    path_parts=s3_path.replace("s3://","").split("/")
    bucket=path_parts.pop(0)
    key="/".join(path_parts)
    return bucket, key

def main():
    st.set_page_config(
        page_title="EMLOv2 - S5 - MMG",
        layout="centered",
        page_icon="🐍",
        initial_sidebar_state="expanded",
    )

    s3_client = boto3.client('s3', region_name='us-east-2')
    st.title("CIFAR10 Classifier")
    
    
    st.subheader("Upload an image to classify it with a Model of Your Choice")

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "png", "jpeg"]
    )
    top_k = st.number_input('Return top k samples', 1,10,10)

    if st.button("Predict"):
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            st.image(image, caption="Uploaded Image", use_column_width=False)
            st.write("")

            try:
                with st.spinner("Downloading Model..."):
                    model_bucket, model_key = split_s3_path(MODEL_PATH)
                    s3_client.download_file(model_bucket,model_key,'model.trace.pt')
                    st.success(f"Model downloaded successfully")
            except:
                st.error("Couldn't fetch the model. Please try again.")
        
            try:
                with st.spinner("Predicting..."):
                    predictions = predict(image, top_k)
                    # get key with highest value
                    st.success(f"Predictions are...")
                    st.markdown(hide_table_row_index, unsafe_allow_html=True)
                    st.table(predictions)
            except:
                st.error("Something went wrong. Please try again.")
                
            try:
                with st.spinner("Flagging Image..."):
                    
                    # get key with highest value
                    # Save the image to an in-memory file
                    in_mem_file = io.BytesIO()
                    image.save(in_mem_file, format=image.format)
                    in_mem_file.seek(0)
                    bucket, key = split_s3_path(FLAGGED_DIR)
                    s3_client.upload_fileobj(in_mem_file, bucket, f'{key}/{uploaded_file.name}')
                    s3_client.download_file(bucket,f'{key}/logs.csv','logs.csv')
                    df = pd.read_csv('logs.csv')
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    row = {'Image': uploaded_file.name, 'Class': predictions['Labels'][0], 'Confidence': predictions['Confidence (%)'][0], 'Date-Time': dt_string}
                    df = pd.concat([df,  pd.DataFrame([row])])
                    csv_buf = io.StringIO()
                    df.to_csv(csv_buf, header=True, index=False)
                    csv_buf.seek(0)
                    s3_client.put_object(Bucket=bucket, Body=csv_buf.getvalue(), Key=f'{key}/logs.csv')
                    st.success(f"Image flagged successfully")
            except:
                st.error("Couldn't flag the predictions. Please try again.")
            
            try:
                with st.spinner("Loading history Image..."):
                    
                    # get key with highest value
                    st.success(f"History of predictions are...")
                    s3_client.download_file(bucket,f'{key}/logs.csv','logs.csv')
                    df = pd.read_csv('logs.csv')
                    # st.markdown(hide_table_row_index, unsafe_allow_html=True)
                    st.table(df)
            except:
                st.error("Couldn't fetch the history. Please try again.")
        else:
            st.warning("Please upload an image.")

    

if __name__ == "__main__":
    main()
