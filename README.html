<h2>EfficientNet-Acute-Promyelocytic-Leukemia (Updated: 2023/04/04)</h2>
<a href="#1">1 EfficientNetV2 Acute-Promyelocytic-Leukemia Classification </a><br>
<a href="#1.1">1.1 Clone repository</a><br>
<a href="#1.2">1.2 Prepare Peripheral Blood Cell dataset</a><br>
<a href="#1.3">1.3 Install Python packages</a><br>
<a href="#2">2 Python classes for Peripheral Blood Cell Classification</a><br>
<a href="#3">3 Pretrained model</a><br>
<a href="#4">4 Train</a><br>
<a href="#4.1">4.1 Train script</a><br>
<a href="#4.2">4.2 Training result</a><br>
<a href="#5">5 Inference</a><br>
<a href="#5.1">5.1 Inference script</a><br>
<a href="#5.2">5.2 Sample test images</a><br>
<a href="#5.3">5.3 Inference result</a><br>
<a href="#6">6 Evaluation</a><br>
<a href="#6.1">6.1 Evaluation script</a><br>
<a href="#6.2">6.2 Evaluation result</a><br>

<h2>
<a id="1">1 EfficientNetV2 Acute-Promyelocytic-Leukemia Classification</a>
</h2>

 This is an experimental Acute-Promyelocytic-Leukemia Image Classification project based on <b>efficientnetv2</b> in <a href="https://github.com/google/automl">Brain AutoML</a>.
<br>
The APL image dataset used here has been taken from the following web site;<br>
<b>
Data record for the article: Deep learning for diagnosis of Acute Promyelocytic Leukemia via recognition of genomically imprinted morphologic features
</b>
<pre>
https://springernature.figshare.com/articles/dataset/Data_record_for_the_article_Deep_learning_for_diagnosis_of_Acute_Promyelocytic_Leukemia_via_recognition_of_genomically_imprinted_morphologic_features/14294675?file=27233798

Citation:
Sidhom, John-William; Siddarthan, Ingharan J.; Lai, Bo Shiun; Luo, Adam; Hambley, Bryan; Bynum, Jennifer; et al. (2021):
 Data record for the article: Deep learning for diagnosis of Acute Promyelocytic Leukemia via recognition of genomically 
 imprinted morphologic features. figshare. Dataset.
https://doi.org/10.6084/m9.figshare.14294675.v1
</pre>
<pre>
Please download the following files:
 blood smear images_Patient00-105.zip
 Images_metadata_table.csv
</pre>
<br>
<br>We use python 3.8 and tensorflow 2.8.0 environment on Windows 11.<br>
<h3>
<a id="1.1">1.1 Clone repository</a>
</h3>
 Please run the following command in your working directory:<br>
<pre>
git clone https://github.com/sarah-antillia/EfficientNet-Acute-Promyelocytic-Leukemia.git
</pre>
You will have the following directory tree:<br>
<pre>
.
├─asset
└─projects
    └─Acute-Promyelocytic-Leukemia
        ├─eval
        ├─evaluation
        ├─inference        
        └─test
</pre>
<h3>
<a id="1.2">1.2 APL dataset</a>
</h3>

1 Splitting <b>images_Patient00-105</b> to <b>Discovery</b> and <b>Validation</b><br> 
We have created <b>AML-API-Images-Patients-Discovery-Validation</b> dataset from the original <b>images_Patient00-105</b>
by using <a href="./projects/APL/Split_Discovery-Validation.py">Split_Discovery-Validation.py</a>, by which we have splitted 
the original <b>images_Patient00-105</b> to <b>Discovery</b> and <b>Validation</b>.
<pre>
>python Split_Discovery-Validation.py
</pre>
The following Discovery and Validation folders will be generated.
<pre>
AML-API-Images-Patients-Discovery-Validation
├─Discovery
│  ├─AML
│  └─APL
└─Validation
    ├─AML
    └─APL
</pre>



2 Splitting <b>Discovery</b> to <b>train</b> and <b>test</b> <br>
Furthermore, we have created <b>AML-APL-Images</b> dataset from the <b>AML-API-Images-Patients-Discovery-Validation/Discovery</b> 
by using <a href="./projects/APL/Split_Discovery.py">Split_Discovery.py</a> script, 
by which we have splitted the master dataset to train and test dataset.<br>
<pre>
>python Split_Discovery.py
</pre> 

The destribution of images in those dataset is the following;<br>
<img src="./projects/APL/_AML-APL-Images_.png" width="720" height="auto"><br>


<pre>
.
├─asset
└─projects
    └─APL
        ├─eval
        ├─evaluation
        ├─inference
        ├─models
        │  └─chief
        ├─AML_APL_Images
        │  ├─test
        │  │  ├─AML
        │  │  └─APL
        │  └─train
        │      ├─AML
        │      └─APL
        └─test
</pre>

<br>


1 Sample images of AML_APL_Images/train/AML:<br>
<img src="./asset/sample_train_images_AML.png" width="840" height="auto">
<br> 

2 Sample images of AML_APLImages/train/APL:<br>
<img src="./asset/sample_train_images_APL.png" width="840" height="auto">
<br> 

<br>


<h3>
<a id="#1.3">1.3 Install Python packages</a>
</h3>
Please run the following commnad to install Python packages for this project.<br>
<pre>
pip install -r requirements.txt
</pre>
<br>

<h2>
<a id="2">2 Python classes for LymphomaClassification</a>
</h2>
We have defined the following python classes to implement our LymphomaClassification.<br>
<li>
<a href="./ClassificationReportWriter.py">ClassificationReportWriter</a>
</li>
<li>
<a href="./ConfusionMatrix.py">ConfusionMatrix</a>
</li>
<li>
<a href="./CustomDataset.py">CustomDataset</a>
</li>
<li>
<a href="./EpochChangeCallback.py">EpochChangeCallback</a>
</li>
<li>
<a href="./EfficientNetV2Evaluator.py">EfficientNetV2Evaluator</a>
</li>
<li>
<a href="./EfficientNetV2Inferencer.py">EfficientNetV2Inferencer</a>
</li>
<li>
<a href="./EfficientNetV2ModelTrainer.py">EfficientNetV2ModelTrainer</a>
</li>
<li>
<a href="./FineTuningModel.py">FineTuningModel</a>
</li>

<li>
<a href="./TestDataset.py">TestDataset</a>
</li>

<h2>
<a id="3">3 Pretrained model</a>
</h2>
 We have used pretrained <b>efficientnetv2-m</b> to train AML model.
Please download the pretrained checkpoint file 
from <a href="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m.tgz">efficientnetv2-m.tgz</a>, expand it, and place the model under our top repository.

<pre>
.
├─asset
├─efficientnetv2-m
└─projects
    └─APM
 ...
</pre>

<h2>
<a id="4">4 Train</a>

</h2>
<h3>
<a id="4.1">4.1 Train script</a>
</h3>
Please run the following bat file to train our APM efficientnetv2 model by using
<b>AML-APLimages/train</b>.
<pre>
./1_train.bat
</pre>
<pre>
rem 1_train.bat
python ../../EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --eval_dir=./eval ^
  --model_name=efficientnetv2-m ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../efficientnetv2-m/model ^
  --optimizer=rmsprop ^
  --image_size=360 ^
  --eval_image_size=360 ^
  --data_dir=./AML-APL-Images/train ^
  --data_augmentation=True ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.0001 ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.4 ^
  --num_epochs=100 ^
  --batch_size=4 ^
  --patience=10 ^
  --debug=True  
</pre>
, where data_generator.config is the following:<br>
<pre>
; data_generation.config

[training]
validation_split   = 0.2
featurewise_center = False
samplewise_center  = False
featurewise_std_normalization=False
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 90
horizontal_flip    = True
vertical_flip      = True
width_shift_range  = 0.05
height_shift_range = 0.05
shear_range        = 0.00
zoom_range         = [0.5, 2.0]

channel_shift_range= 10
brightness_range   = [80,100]
data_format        = "channels_last"
</pre>

<h3>
<a id="4.2">4.2 Training result</a>
</h3>

This will generate a <b>best_model.h5</b> in the models folder specified by --model_dir parameter.<br>
Furthermore, it will generate a <a href="./projects/APL/eval/train_accuracies.csv">train_accuracies</a>
and <a href="./projects/APL/eval/train_losses.csv">train_losses</a> files
<br>
Training console output:<br>
<img src="./asset/train_at_epoch_23_0404.png" width="740" height="auto"><br>
<br>
Train_accuracies:<br>
<img src="./projects/APL/eval/train_accuracies.png" width="640" height="auto"><br>

<br>
Train_losses:<br>
<img src="./projects/APL/eval/train_losses.png" width="640" height="auto"><br>

<br>
<h2>
<a id="5">5 Inference</a>
</h2>
<h3>
<a id="5.1">5.1 Inference script</a>
</h3>
Please run the following bat file to infer the skin cancer lesions in test images by the model generated by the above train command.<br>
<pre>
./2_inference.bat
</pre>
<pre>
rem 2_inference.bat
python ../../EfficientNetV2Inferencer.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.4 ^
  --image_path=./test/*.jpg ^
  --eval_image_size=360 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --infer_dir=./inference ^
  --debug=False 
</pre>
<br>
label_map.txt:
<pre>
AML
APL
</pre>
<br>
<h3>
<a id="5.2">5.2 Sample test images</a>
</h3>

Sample test images generated by <a href="./projects/AML/create_test_dataset.py">create_test_dataset.py</a> 
from <a href="./projects/APL/AML-APL-Images/test">Lymphoma/test</a>.
<br>
<img src="./asset/test.png" width="840" height="auto"><br>


<br>
<h3>
<a id="5.3">5.3 Inference result</a>
</h3>
This inference command will generate <a href="./projects/Acute-Promyelocytic-Leukemia/inference/inference.csv">inference result file</a>.
<br>At this time, you can see the inference accuracy for the test dataset by our trained model is very low.
More experiments will be needed to improve accuracy.<br>

<br>
Inference console output:<br>
<img src="./asset/inference_at_epoch_23_0404.png" width="740" height="auto"><br>
<br>

Inference result (<a href="./projects/APL/inference/inference.csv">inference.csv</a>):<br>
<img src="./asset/inference_at_epoch_23_0404_csv.png" width="640" height="auto"><br>
<br>
<h2>
<a id="6">6 Evaluation</a>
</h2>
<h3>
<a id="6.1">6.1 Evaluation script</a>
</h3>
Please run the following bat file to evaluate <a href="./projects/Acute-Promyelocytic-Leukemia/Resampled_AML_Images/test">
Malaris_Cell_Images/test</a> by the trained model.<br>
<pre>
./3_evaluate.bat
</pre>
<pre>
rem 3_evaluate.bat
python ../../EfficientNetV2Evaluator.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --data_dir=./AML-APL-Images/test ^
  --evaluation_dir=./evaluation ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.4 ^
  --eval_image_size=360 ^
  --mixed_precision=True ^
  --debug=False 
</pre>
<br>

<h3>
<a id="6.2">6.2 Evaluation result</a>
</h3>

This evaluation command will generate <a href="./projects/APL/evaluation/classification_report.csv">a classification report</a>
 and <a href="./projects/APL/evaluation/confusion_matrix.png">a confusion_matrix</a>.
<br>
<br>
Evaluation console output:<br>
<img src="./asset/evaluate_at_epoch_23_0404.png" width="740" height="auto"><br>
<br>

<br>
Classification report:<br>
<img src="./asset/classification_report_at_epoch_23_0404.png" width="740" height="auto"><br>
<br>
Confusion matrix:<br>
<img src="./projects/APL/evaluation/confusion_matrix.png" width="740" height="auto"><br>

<br>
<h3>
References
</h3>
<b>1. Deep learning for diagnosis of Acute Promyelocytic Leukemia via recognition of genomically 
 imprinted morphologic features. figshare. Dataset</b><br>
<pre>
https://springernature.figshare.com/articles/dataset/Data_record_for_the_article_Deep_learning_for_diagnosis_of_Acute_Promyelocytic_Leukemia_via_recognition_of_genomically_imprinted_morphologic_features/14294675?file=27233798

Sidhom, John-William; Siddarthan, Ingharan J.; Lai, Bo Shiun; Luo, Adam; Hambley, Bryan; Bynum, Jennifer; et al. (2021):
 Data record for the article: Deep learning for diagnosis of Acute Promyelocytic Leukemia via recognition of genomically 
 imprinted morphologic features. figshare. Dataset.
https://doi.org/10.6084/m9.figshare.14294675.v1
</pre>

<b>2. Detection of acute promyelocytic leukemia in peripheral blood and bone marrow with annotation-free deep learning</b><br>
Petru Manescu, Priya Narayanan,Christopher Bendkowski,Muna Elmi,Remy Claveau,Vijay Pawar,Biobele J. Brown,<br>
Mike Shaw, Anupama Rao, and Delmiro Fernandez-Reyes<br>
<pre>
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9925435/
</pre>

<b>3. Deep learning for diagnosis of acute promyelocytic leukemia via recognition of genomically imprinted morphologic features</b><br>
John-William Sidhom, Ingharan J. Siddarthan, Bo-Shiun Lai, Adam Luo, Bryan C. Hambley, Jennifer Bynum, Amy S.<br>
Duffield, Michael B. Streiff, Alison R. Moliterno, Philip Imus, Christian B. Gocke, Lukasz P. Gondek, Amy E. DeZern,<br>
Alexander S. Baras, Thomas Kickler, Mark J. Levis & Eugene Shenderov<br>
<pre>
https://www.nature.com/articles/s41698-021-00179-y
</pre>
