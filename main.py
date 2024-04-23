import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from segmentation_models_pytorch import UnetPlusPlus
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import os

IMAGE_PATH = Path("image")
IMAGE_PATH_LIST = list(IMAGE_PATH.glob("*.png"))
IMAGE_PATH_LIST = sorted(IMAGE_PATH_LIST)

print(f'Total Images = {len(IMAGE_PATH_LIST)}')

TEMP_PATH = Path("temp")
TEMP_PATH_LIST = list(TEMP_PATH.glob("*.png"))
TEMP_PATH_LIST = sorted(TEMP_PATH_LIST)

print(f'Total Temps = {len(TEMP_PATH_LIST)}')

data = pd.DataFrame({'Image':IMAGE_PATH_LIST, 'Temp':TEMP_PATH_LIST})
data.head()

data = data.reset_index(drop=True)

class CustomImageMaskDataset(Dataset):
    def __init__(self, data: pd.DataFrame, image_transforms: transforms, temp_transforms: transforms):
        self.data = data
        self.image_transforms = image_transforms
        self.temp_transforms = temp_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        image = Image.open(image_path).convert("RGB")
        image = self.image_transforms(image)

        temp_path = self.data.iloc[idx, 1]
        temp = Image.open(temp_path).convert("RGB")
        temp = self.temp_transforms(temp)

        # Преобразование в тензоры
        image_tensor = image.clone().detach()
        temp_tensor = temp.clone().detach()

        # Конкатенация изображения
        concatenated_data = torch.cat((image_tensor, temp_tensor), dim=0)

        return concatenated_data

# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device
print(device)


def predictions(test_dataloader: torch.utils.data.DataLoader):
    loaded_model = UnetPlusPlus(encoder_name="resnext50_32x4d",
                                encoder_weights="imagenet",
                                in_channels=6,
                                classes=1)

    checkpoint = torch.load("icemodel.pth")

    loaded_model.load_state_dict(checkpoint)

    loaded_model.to(device)

    loaded_model.eval()

    pred_mask_test = []

    with torch.inference_mode():
        for X in tqdm(test_dataloader):
            X = X.to(device, dtype=torch.float)
            logit_mask = loaded_model(X)
            prob_mask = logit_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
            pred_mask_test.append(pred_mask.detach().cpu())

    pred_mask_test = torch.cat(pred_mask_test)

    return pred_mask_test

if __name__ == '__main__':
    RESIZE = (224, 224)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    image_transforms = transforms.Compose([transforms.Resize(RESIZE),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = MEAN,
                                                                std = STD)])

    RESIZE = (224, 224)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    temp_transforms = transforms.Compose([transforms.Resize(RESIZE),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = MEAN,
                                                                std = STD)])

    BATCH_SIZE = 1
    NUM_WORKERS = 1

    test_dataset = CustomImageMaskDataset(data,
                                          image_transforms,
                                          temp_transforms )

    test_dataloader = DataLoader(dataset = test_dataset,
                                 batch_size = BATCH_SIZE,
                                 shuffle = False,
                                 num_workers = NUM_WORKERS)

    pred_mask_test = predictions(test_dataloader)

    # Создание директории для сохранения масок, если она не существует
    output_dir = "predicted_masks"
    os.makedirs(output_dir, exist_ok=True)

    # Сохранение каждой маски из pred_mask_test в отдельный PNG-файл
    for i, mask in enumerate(pred_mask_test):
        # Преобразование тензора маски обратно в изображение PIL
        mask_array = mask.squeeze().cpu().numpy().astype('uint8')  # преобразование тензора в массив numpy
        mask_image = Image.fromarray(
            mask_array * 255)  # создание изображения из массива (преобразование обратно в 0-255)

        # Составление пути для сохранения
        filename = f"mask_{i}.png"  # формирование имени файла
        filepath = os.path.join(output_dir, filename)  # формирование полного пути к файлу

        # Сохранение изображения в формате PNG
        mask_image.save(filepath)

        print(f"Saved mask {i} to {filepath}")

    print("All masks saved successfully.")






