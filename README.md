

Description
-----
San Diego Crime Visualization is a project that generate a html Google Map describing the crime frequency. The data is collected from "http://www.sandag.org/index.asp?classid=14&subclassid=21&projectid=446&fuseaction=projects.detail". 

The user can also enter an address in string format "9500 Gilman Drive" to get a map around that area.

Links
-----
Slides: https://docs.google.com/presentation/d/1OLELE8NHs6DHJo2HIM2QdGfhkBhOPcE-ZgxQJUCCY0c/edit#slide=id.p

Members:
-----
Yuming Qiao (y1qiao@ucsd.edu)

Tim Miller (timtheviolinist@gmail.com)

Taylor Hua (t5hua@ucsd.edu)

How to run:
-----
To run the code, you need to install (python 2.7)
1. numpy
2. uszipcode - pip install uszipcode
3. gmplot - please install from the github repo directly, the pip install will only give you an outdated version.
- pip install git+https://github.com/vgm64/gmplot.git
4. The code uses google service to work, the service key was removed for privacy, the user need to have a valid google service key to run the code.
5. The full_project_classes.py collects all the jupyter notebook codes, cleans up all the code and is well commented. 

6.The html heatmap code is in crime_last_week_heatmap_plot.ipynb. 

