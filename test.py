from classes import Img2Xlsx
# pars_path=input('path to parser (default ./utils/parser.pt):')
# pars_dir=input('path to parser directory (default ./utils/pars_img):')
pars_img=input('path to img:')
table_name=input('path/name result table:')
model=Img2Xlsx()
model.convert(pars_img, table_name)