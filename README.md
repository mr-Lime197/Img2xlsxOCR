# Img2xlsxOCR
### Img2xlsxOCR - a repository that provides the python class `Img2Xlsx` for recognizing files of a certain structure and forming an xlsx document based on this data.

## Minimum technical requirements
|VRAM|Disk space|Additional streds|
|:-:|:-------------------:|:----------:|
|16 гб| 20 гб|VPN

## Installing
### requiroments.txt
installing requiroments file with library
``` pip install -r requirements ```

## Usage
clone this git repository into your project

```git clone https://github.com/mr-Lime197/Img2xlsxOCR.git ```

in the file classes.py the main class `Img2Xlsx` is located. **IMPORTANT: A VPN CONNECTION IS REQUIRED TO WORK.**
### Sample usage
```
from classes import Img2Xlsx
pars_img=input('path to img:')
table_name=input('path/name result table:')
model=Img2Xlsx()
model.convert(pars_img, table_name)
```
* pars_img - path to your image with document
* table_name - path/name to your result xlsx file