# Image-Classification

This project demonstrates the use of a Vision Transformer (ViT) model for image classification, utilizing the ```google/vit-base-patch16-224``` pre-trained model from Hugging Face. The workflow includes data preprocessing, model fine-tuning, evaluation, and visualization of a confusion matrix for classification results.

## Prerequisite
```pip install torch torchvision transformers datasets pandas numpy scikit-learn tqdm matplotlib seaborn Pillow requests```

## Dataset
The dataset should be a CSV file (```train.csv```) with the following columns:
* image_url: URLs of the images to be classified.
* label: The corresponding label for each image.

## Workflow
* Data Preprocessing: The images are downloaded from URLs, resized to 224x224, normalized using the pre-trained model's mean and standard deviation, and then converted into tensors. Labels are also encoded using ```LabelEncoder```.

* Model Initialization: The pre-trained ```google/vit-base-patch16-224``` model is loaded using Hugging Face's AutoModelForImageClassification. The number of output labels is dynamically adjusted based on the dataset.

* Dataset Preparation: The dataset is split into training (80%), validation (10%), and testing (10%) sets. 

* Training: The model is fine-tuned using ```Trainer``` with the following hyperparameters:
>* Learning rate: 3e-5
>* Epochs: 2
>* Weight decay: 0.01
>* Batch size: 16
>* Evaluation is performed at each epoch based on the F1 score.
* Evaluation: The model is evaluated on the validation set using accuracy and F1 score. After training, the model is saved to the ```trained_model/``` directory.

* Inference and Confusion Matrix: After training, the trained model is used for inference on the test set. Predictions are made using Hugging Face's ```pipeline```. The predicted and true labels are compared, and a confusion matrix is computed and visualized.