# CeMiA
[![N|Solid](https://raw.githubusercontent.com/Mitogenie/CeMiA/master/misc/KLab.png)](https://mic.med.virginia.edu/kashatus/)

### Cellular Mitochondrial Analyzer 

![Build Status](https://raw.githubusercontent.com/Mitogenie/CeMiA/master/misc/ver.png)
##### CeMiA is a set of tools to enable high-throughput analysis of mitochondrial network morphology.

  - ##### Cell Catcher
  - ##### Mito Miner
  - ##### MiA
  - ##### Nuc Adder
  

###### User Interface
All the tools in CeMiA toolkit offer interactive, semi graphical user interface through Jupyter notebooks.
###### Processing
All the tools analyze one cell at a time. However, Cell Catcher and Mito Miner, benefit from Multi-Threading, where applicable, to enhance the performance.
###### Requirements
  - CeMiA toolkit is developed in Jupyter notebooks using Python 3. Since it benefits from a wide range of Python libraries, along with Jupyter notebooks, we suggest installing Anaconda distribution of Python (V 3.7). --> [Download Anaconda](https://www.anaconda.com/distribution/)
  - While Anaconda installation takes care of most of the library requirements for CeMiA, there is only one more library (OpenCV) that needs to be installed, which can be done through the command line. (You just need to copy and run the following command. (without the $))
    - Windows: Use Anaconda Prompt.
    - MacOS, Linux: Use Terminal.
```sh
$ pip install opencv-python==3.4.2.17
```
  - All the tools (Jupyter notebook files) in CeMiA package, depend on cemia55s.py to run. This module includes all the functions used in the develepment of these tools. This file should be in the same folder as the jupyter notebook you are running.
  
### CeMiA Toolkit

#### Cell Catcher
Cell Catcher is a tool designed to automatically detect, separate, and isolate individual cells from 2d multi-cell images. This tool uses the statistical distribution of mitochondria and nuclei across the image to separate individual cells from the images and export them as single-cell images.

###### What to know before use
- Input: Standard RGB Tiff images
    - Multi-cell fluorescence images
        - Number of stains: 2 
            - Nuclei should be stained with DAPI
- Output: Single-cell fluorescence images (Standard RGB Tiff)
    - Output images can used as input for Mito Miner to extract their mitochondrial network, or they can be independently used to train the desired ML or NN models where appropriate. 

###### Instructions
- Run Cell_Catcher.ipynb using Jupyter notebook.
    - You may find this video on Jupyter notebooks very helpful: [Watch Here](https://youtu.be/HW29067qVWk)
        - Note: We are not affiliated with the owner of the above video. We just found this video on Jupyter notebooks very helpful, There are plenty of great tutorials about this subject, and you may use any source your prefer.
- The Jupyter notebook file provides the users with step-by-step instructions to analyze their data.

###### Development
- 500+ images were used to develop and test Cell Catcher.

#### Mito Miner
Mito Miner is a tool to segment mitochondrial netwrok in the cells. It uses the statistical distribution of pixel intensities across the mitochondrial network to detect and remove background noise from the cell and segment the mitochondrial network. Additionally, this tool can further improve the accuracy of the mitochondrial network segmentation through an optional adaptive correction, which takes the variation in the efficiency of fluorescence staining across each cell into account to enhance mitochondrial segmentation. 
###### What to know before use
- Input: Standard RGB Tiff images
    - Single-cell fluorescence images
        - Number of stains: 2 
            - Nuclei should be stained with DAPI
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
MiA uses the binarized mitochondrial network to perform greater than 100 mitochondria-level and cell-level morphometric measurements. 
###### What to know before use
- Input: Single-cell binary images of mitochondrial network
- Output: Tabular data (TSV format), including aggregate(per cell, and per mitochondrion where applicable) and raw (per mitochondrion) measurements. 
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
Cell Catcher, and Mito Miner require RGB images of the cells, where nuclei are stained with DAPI. They use nuceli boundary to estimate the background intensity in each cell. However, in some images, nuclei staining is not available. By using Nuc Adder, you can transfrom your images and adapt them for tools in CeMiA. Nuc Adder, simply adds a circle as a synthetic nuclei to gather mitochondrial background info from each cell.

###### What to know before use
- Input: Standard RGB Tiff images
    - Single- or Multi-cell fluorescence images
        - Number of stains: 1 
            - The stanied channel should be red or green.
- Output: Multi- or Single-cell fluorescence images (Standard RGB Tiff)
    - Output images with synthetic nuclei can used as input for Mito Miner or Cell Catcher. 

###### Instructions
- Run Nuc_Adder.ipynb using Jupyter notebook.
    - You may find this video on Jupyter notebooks very helpful: [Watch Here](https://youtu.be/HW29067qVWk)
        - Note: We are not affiliated with the owner of the above video. We just found this video on Jupyter notebooks very helpful, There are plenty of great tutorials about this subject, and you may use any source your prefer.
- The Jupyter notebook file provides the users with step-by-step instructions to analyze their data.

###### Development
- 10+ images were used to develop and test Cell Catcher.

#### Nomenclature of Features
##### Mitochondria Level Measurements
- These features have the following format: mito_<feature name> and are raw measurements.
    - Examples:
        - area of individual mitochondrion in the cell: mito_area
##### Cell Level Measurements
###### Mitochondrial aggregate measurements
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

###### Cell level measurements based on mitochondria distribution
- These features have the following format: cell_<feature name> and are aggregate measurements.
    - Examples:
        - Fractal dimension of the mitochondrial network of the cell: cell_network_fractal_dimension

Details of all the features measured by MiA can be found here: [Features Dictionary.txt](https://github.com/Mitogenie/CeMiA/blob/master/Features%20Dictionary.txt)

#### Tree Structure of the files and folders

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

##### Different Apps Input/Output Folders:
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

#### Libraries used for development (A/Z)
- copy
- cv2
- datetime
- math
- matplotlib.pyplot
- numpy
- os
- pandas
- random
- scipy
- shutil
- skimage

