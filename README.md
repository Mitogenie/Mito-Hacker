# CeMiA

[![N|Solid](https://raw.githubusercontent.com/Mitogenie/misc/master/misc/KLab.png)](https://mic.med.virginia.edu/kashatus/)

## Cellular Mitochondrial Analyzer

![Build Status](https://raw.githubusercontent.com/Mitogenie/misc/master/misc/CeMiA_ver.png)
#### CeMiA is a set of tools to enable high-throughput analysis of mitochondrial network morphology.

  - ### Cell Catcher
    - Cell Catcher is a tool designed to automatically detect, separate, and isolate individual cells from 2d multi-cell images. This tool uses the statistical distribution of mitochondria and nuclei across the image to separate individual cells from the images and export them as single-cell images.

  - ### Mito Miner
    - Mito Miner is a tool to segment mitochondrial network in the cells. It uses the statistical distribution of pixel intensities across the mitochondrial network to detect and remove background noise from the cell and segment the mitochondrial network. Additionally, this tool can further improve the accuracy of the mitochondrial network segmentation through an optional adaptive correction, which takes the variation in the efficiency of fluorescence staining across each cell into account to enhance mitochondrial segmentation.

  - ### MiA
    - MiA uses the binarized mitochondrial network to perform greater than 100 mitochondria-level and cell-level morphometric measurements.

  - ### Nuc Adder (Optional)
    - Cell Catcher, and Mito Miner require 2d RGB images of the cells, where nuclei are stained with DAPI (or any blue fluorescent dye). They use nuclei boundaries to estimate the background intensity in each cell. However, in some images, nuclei staining is not available. By using Nuc Adder, you can transfrom your images and adapt them for tools in CeMiA. Nuc Adder simply adds a circle as a synthetic nucleus to gather mitochondrial background info from each cell.

[Read more about these tools here](#more-on-cemia-tools)

##### User Interface
All the tools in CeMiA toolkit offer interactive, semi graphical user interface through Jupyter notebooks.

##### Software Requirements
Follow these instructions if you do not have Anaconda or Jupyter notebooks installed on your computer.
</br>
  - CeMiA toolkit is developed in Jupyter notebooks using Python 3. Since it benefits from a wide range of Python libraries, along with Jupyter notebooks, we suggest installing Anaconda distribution of Python (V 3.7). --> [Download Anaconda](https://www.anaconda.com/distribution/)
You may follow the rest of the instructions if you do not already have OpenCV installed on your computer.
  - While Anaconda installation takes care of most of the library requirements for CeMiA, there is only one more libary (OpenCV) that needs to be installed, which can be achieved through the command line. (You just need to copy and run the following command. (without the $))
    - Windows: Use Anaconda Prompt.
    - MacOS, Linux: Use Terminal.
```sh
$ pip install opencv-python==3.4.2.17
```
  - It is important to install this specific version of OpenCV for compatibility.
  - All the tools (Jupyter notebook files) in CeMiA package, depend on cemia55s.py to run. This module includes all the functions used in the develepment of these tools. This file should be in the same folder as the jupyter notebook you are running.

[Go back to the top](#cellular-mitochondrial-analyzer)

### Where to start your analysis?
##### 1) Download the files on your computer
Once you have satistfied the requirements, you can start your analysis by downloading or cloning this repository on your computer. The simplest way is to download the whole directory by pressing the green button (top right) and download the ZIP file.
##### 2) Decide on the proper tool to use at each step
The following Flow chart helps you to choose the best tool based on the data you have.
![Flow Chart](https://raw.githubusercontent.com/Mitogenie/misc/master/misc/CeMiA_flowchart.png)

##### 3) Follow the step-by-step instructions in the tool you are using

[Go back to the top](#cellular-mitochondrial-analyzer)

### More on CeMiA Tools

#### Cell Catcher

###### What to know before use
- Input: Standard 2d RGB Tiff images
    - Multi-cell fluorescence images
        - Number of stains: 2
            - Nuclei should be stained with DAPI (any blue fluorescent dye)
- Output: Single-cell fluorescence images (Standard 2d RGB Tiff)
    - Output images can used as input for Mito Miner to extract their mitochondrial network, or they can be independently used to train the desired ML or NN models where appropriate.

###### Instructions
- Run Cell_Catcher.ipynb using Jupyter notebook.
    - You may find this video on Jupyter notebooks very helpful: [Watch Here](https://youtu.be/HW29067qVWk)
        - Note: We are not affiliated with the owner of the above video. We just found this video on Jupyter notebooks very helpful, There are plenty of great tutorials about this subject, and you may use any source your prefer.
- The Jupyter notebook file provides the users with step-by-step instructions to analyze their data.

###### Development
- 500+ images were used to develop and test Cell Catcher.

#### Mito Miner

###### What to know before use
- Input: Standard 2d RGB Tiff images
    - Single-cell fluorescence images
        - Number of stains: 2
            - Nuclei should be stained with DAPI (or other blue fluorescent dye)
- Output: Single-cell binary images of mitochondrial network
    - Output images can used as input for MiA to quantify their mitochondrial network, or they can be independently used to train the desired ML or NN models where appropriate.
###### Instructions:
- Run Mito_Miner.ipynb using Jupyter notebook.
    - You may find this video on Jupyter notebooks very helpful: [Watch Here](https://youtu.be/HW29067qVWk)
        - Note: We are not affiliated with the owner of the above video. We just found this video on Jupyter notebooks very helpful, There are plenty of great tutorials about this subject, and you may use any source your prefer.
- The Jupyter notebook file provides the users with step-by-step instructions to analyze their data.
###### Development
- 7500+ images were used to develop and test Mito Miner.

#### MiA (Mitochondrial Analyzer)

###### What to know before use
- Input: Single-cell binary images of mitochondrial network
- Output: Tabular data (TSV format), including aggregate (per cell, and per mitochondrion where applicable) and raw (per mitochondrion) measurements.
    - Why raw data along with aggregate measurements?
        - Flexibility! By providing raw data, users can limit or aggregate their data using their own criteria on each mitochondrial, or sub mitochondrial measurements, which can provide additional insight based on their specific applications.
###### Instructions:
- Run MiA.ipynb using Jupyter notebook.
    - You may find this video on Jupyter notebooks very helpful: [Watch Here](https://youtu.be/HW29067qVWk)
        - Note: We are not affiliated with the owner of the above video. We just found this video on Jupyter notebooks very helpful, There are plenty of great tutorials about this subject, and you may use any source your prefer.
- The Jupyter notebook file provides the users with step-by-step instructions to analyze their data.
###### Development
- 4500+ images were used to develop and test MiA.

#### Nuc Adder (Optional Tool)

###### What to know before use
- Input: Standard 2d RGB Tiff images
    - Single- or Multi-cell fluorescence images
        - Number of stains: 1
            - The stanied channel should be red or green.
- Output: Multi- or Single-cell fluorescence images (Standard 2d RGB Tiff)
    - Output images with synthetic nuclei can used as input for Mito Miner or Cell Catcher.

###### Instructions
- Run Nuc_Adder.ipynb using Jupyter notebook.
    - You may find this video on Jupyter notebooks very helpful: [Watch Here](https://youtu.be/HW29067qVWk)
        - Note: We are not affiliated with the owner of the above video. We just found this video on Jupyter notebooks very helpful, There are plenty of great tutorials about this subject, and you may use any source your prefer.
- The Jupyter notebook file provides the users with step-by-step instructions to analyze their data.

###### Development
- 10+ images were used to develop and test Cell Catcher.

##### Processing
In the current version, all the tools analyze one cell at a time using the CPU. Currently we have a beta version of Cell Catcher and Mito Miner that benefit from Multi-Threading, where applicable, to enhance the performance. Once they are fully tested, we will release them in this repository.

[Go back to the top](#cellular-mitochondrial-analyzer)

### Nomenclature of Features

What does each measured feature mean?

#### Mitochondria Level Measurements
- These features have the following format: mito_<feature name> and are raw measurements.
    - Examples:
        - area of individual mitochondrion in the cell: mito_area

#### Cell Level Measurements
There are two categories of cell Level measurements.

#### Mitochondrial aggregate measurements
- These are cell-level measurements that are aggregates of raw mitochondrial-level measurements.
    - cell_mean_mito_<feature>: Average of the feature among all the mitochondria in the cell.
        - Example:
            - cell_mean_mito_area: Average of the area among all the mitochondria in the cell.

    - cell_median_mito_<feature>: Median of the feature among all the mitochondria in the cell.
        - Example:
            - cell_median_mito_area: Median of the area among all the mitochondria in the cell.
    - cell_std_mito_<feature>: Standard Deviation of the feature among all the mitochondria in the cell.
        - Example:
            - cell_std_mito_area: Standard deviation of the area among all the mitochondria in the cell.

#### Cell level measurements based on mitochondria distribution
- These features have the following format: cell_<feature name> and are aggregate measurements.
    - Examples:
        - Fractal dimension of the mitochondrial network of the cell: cell_network_fractal_dimension

Details of all the features measured by MiA can be found here: [Features Dictionary.txt](https://github.com/Mitogenie/misc/blob/master/misc/CeMiAFeaturesDictionary.txt)

[Go back to the top](#cellular-mitochondrial-analyzer)

### Tree Structure of the files and folders (where are the I/O of different apps?)

```sh
your_project_folder/ (Name of you project folder)
├── *.tif (Original images in the project)
├── output/
│     ├── to_analyze/
│     │          ├── *.tif  (Cell Catcher Output/Mito Miner input files)
│     ├── to_discard/
│     │          ├── *.tif  (These files did not pass the initial quality check by Cell Catcher)
│     └── processed/
│                └── single_cells_binary/
│                                 ├── *_binarized.tif (Mito Miner output/MiA input files)
│                                 └── *.tif  (Single cells isolated by Cell Catcher)
│
├── transformed/
│             ├── *.tif  (Cells with synthetic nucleii) (Optional Tool)
├── cell_catcher_params.csv
├── mito_miner_params.csv
├── Mia_output_file.csv (Named by user)
└── cell_catcher_temp/ (Can optionally be deleted after using Cell Catcher)
```

#### Different Apps Input/Output Folders:
- Cell Catcher:
  - Input: your_project_folder/
  - Output: your_project_folder/output/to_analyze
- Mito Miner:
  - Input: your_project_folder/output/to_analyze
  - Output: your_project_folder/output/processed/single_cells_binary/
- MiA
  - Input: your_project_folder/output/processed/single_cells_binary/
  - Output: your_project_folder/
- Nuc Adder
  - Input: your_project_folder/
  - Output: your_project_folder/transformed

[Go back to the top](#cellular-mitochondrial-analyzer)

### List of the libraries we used for development (A/Z)
- copy
- cv2
- datetime
- ipywidgets
- math
- matplotlib.pyplot
- numpy
- os
- pandas
- random
- scipy
- shutil
- skimage

[Go back to the top](#cellular-mitochondrial-analyzer)

### A note on imaging

Images should be acquired at a consistent pixel size. The software will convert all images to standard 1024 x 1024 dimensions for internal analysis, then adjust the measurements to reflect the conversion in original image size. All the measurements are in pixel units, the user can apply a scaling factor to their data based on their own pixel size to convert measurements to microns. The user should be cautious about direct comparisons of pixel data from images of different sizes. While square images are preferred, there is no limit on size or aspect ratio of the images.

[Go back to the top](#cellular-mitochondrial-analyzer)

