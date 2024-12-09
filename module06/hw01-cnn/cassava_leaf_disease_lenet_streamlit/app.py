import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

idx2label = {
    0: "cbb",
    1: "cbsd",
    2: "cgm",
    3: "cmd",
    4: "healthy",
}


class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=5, padding="same"
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 35 * 35, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.avgpool1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.avgpool2(outputs)
        outputs = F.relu(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc_1(outputs)
        outputs = self.fc_2(outputs)
        outputs = self.fc_3(outputs)
        return outputs


@st.cache_resource
def load_model(model_path, num_classes=5):
    lenet_model = LeNetClassifier(num_classes)
    lenet_model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))
    )
    lenet_model.eval()
    return lenet_model


model = load_model("lenet_model_cassava.pt")


def inference(image, model):
    img_size = 150
    img_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    img_new = img_transform(image)
    img_new = torch.unsqueeze(img_new, 0)
    with torch.no_grad():
        predictions = model(img_new)
    preds = nn.Softmax(dim=1)(predictions)
    p_max, yhat = torch.max(preds.data, 1)
    return p_max.item() * 100, yhat.item()


def main():
    st.title("Cassava Leaf Disease Classfication")
    st.subheader("Model: LeNet. Dataset: Cassava Leaf Disase")
    option = st.selectbox(
        "How would you like to give the input?",
        ("Upload Image File", "Run Example Image"),
    )
    if option == "Upload Image File":
        file = st.file_uploader("Please upload an image", type=["jpg", "png"])
        if file is not None:
            image = Image.open(file)
            p, idx = inference(image, model)
            label = idx2label[idx]
            st.image(image)
            st.success(
                f"The uploaded image is of the {label} with {p:.2f} % probability."
            )

    elif option == "Run Example Image":
        image = Image.open("demo_cbsd.jpg")
        p, idx = inference(image, model)
        label = idx2label[idx]
        st.image(image)
        st.success(f"The image is of the {label} with {p:.2f} % probability.")


if __name__ == "__main__":
    main()
