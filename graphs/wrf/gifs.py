import os, sys
import imageio
import glob
###### Namelist ########
# 0-hfx                  #
# 1-lh                   #
# 2-pressure             #
# 3-rain                 #
# 4-temperature          #
# 5-vapor                #
# 6-wind                 #
# 7-swdown               #
########################
"""
os.chdir(dataDir)
fileList = glob.glob('*.png')
fileList.sort()
file = open('temperature.txt', 'w')
for item in fileList:
    file.write("%s\n" % item)
file.close()
"""
dataDir = ['C:\\Users\\BrunoM\\Documents\\working now\\modelo\\HFX','C:\\Users\\BrunoM\\Documents\\working now\\modelo\\LH',
           'C:\\Users\\BrunoM\\Documents\\working now\\modelo\\pressure','C:\\Users\\BrunoM\\Documents\\working now\\modelo\\rain',
           'C:\\Users\\BrunoM\\Documents\\working now\\modelo\\temperature','C:\\Users\\BrunoM\\Documents\\working now\\modelo\\vapor',
           'C:\\Users\\BrunoM\\Documents\\working now\\modelo\\wind','C:\\Users\\BrunoM\\Documents\\working now\\modelo\\SWDOWN']

#dataDir1='C:\\Users\\BrunoM\\Documents\\working now\\modelo\\temperature'

def generateGifs(dataDir,var):
    images = []
    for file_name in os.listdir(dataDir):
        if file_name.endswith('.png'):
            file_path = os.path.join(dataDir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(var+'.gif', images, duration=0.5)

generateGifs(dataDir[0],'hfx')
generateGifs(dataDir[1],'lh')
generateGifs(dataDir[2],'pressure')
generateGifs(dataDir[3],'rain')
generateGifs(dataDir[4],'temperature')
generateGifs(dataDir[5],'vapor')
generateGifs(dataDir[6],'wind')
generateGifs(dataDir[7],'swdown')