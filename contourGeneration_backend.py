# Module for Contour Generation in Liquid Crystal Software
# Gets user input from the UI and generates contours appropriately

import cv2
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import random
from skimage import measure
from random import randrange
import PIL
from PIL import Image, ImageTk, ImageDraw
import os

class LC_Contour:
    def __init__(self, w, h, c, image_path, kernel_size, closed_contour_th, plot_interval, cropped_tl, cropped_br):
        #Intialize contour object with user selected options and generate the contours
        self.w = w
        self.h = h
        self.color = c
        self.path = image_path
        self.ext = os.path.splitext(self.path)[-1]
        self.contour_area_th = closed_contour_th
        locList = []
        locList.append([cropped_tl[0],cropped_tl[1]])
        locList.append([cropped_br[0],cropped_br[1]])
        self.boundstart = locList
        self.imageReadResize()
        self.grayImage = cv2.cvtColor(self.imageResized, cv2.COLOR_RGB2GRAY)
        image_resized = self.imageReconstruct()
        image_resized_gb = cv2.GaussianBlur(image_resized, kernel_size, 0) #gaussian blur added to grayscale image
        cv2.imwrite('Image size test.png',image_resized_gb)
        if self.color == "red" or self.color == "green" or self.color == "blue": #depending on which option the user has chosen. Is it RGB or HSV and then process normalized colors based on that.
            normalized_color = self.normalizeRgb(image_resized_gb)
        elif self.color == "hue" or self.color == "saturation" or self.color == "value":
            normalized_color = self.normalizeHsv(image_resized_gb)
        contours, isovals, centroids, endpoints = self.contourDetection(image_resized_gb, normalized_color, plot_interval) #function to detect contours
        contour_fig = self.drawFigure(contours, isovals, centroids, normalized_color)
        self.fig2data(contour_fig)
        cv2.imwrite('self cv fig.png',self.cv)
        self.addText(isovals, centroids, endpoints)
        self.contour.save('Final Contour Image '+self.color+self.ext)

    def saveOptions(self, clicked_res, clicked_color):
        res = clicked_res.get()
        w = res.split("x")
        self.w = int(w[0])
        self.h = int(w[1])
        self.color = clicked_color.get()

    def show(self):
        myLabel = Label(root, text = variable.get.pack()).pack()
        self.color = myLabel
        
    def saveInputs(self, color_selected):
        # Saves the inputs from the UI as class global variables
        self.color = color_selected
        
    def imageReadResize(self):
        # Resizes the image
        img = PIL.Image.open(self.path)
        img = img.resize((self.w, self.h), PIL.Image.LANCZOS)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self.imageResized = img_cv
        
    def imageReconstruct(self):
        # Crops and returns the image ROI
        img = self.imageResized
        new_img = img[self.boundstart[0][1]:self.boundstart[1][1], self.boundstart[0][0]:self.boundstart[1][0]]
        new_img_resized = cv2.resize(new_img, (self.w, self.h), interpolation= cv2.INTER_LINEAR)
        cv2.imwrite("cropped image.png", new_img)
        return new_img_resized
        
    def normalizeRgb(self, image):
        # Normalize RGB color
        if self.color == "red":
            red = image[:,:,2]
            green = image[:,:,1]
            blue = image[:,:,0]
        elif self.color == "green":
            red = image[:,:,1]
            green = image[:,:,2]
            blue = image[:,:,0]
        elif self.color == "blue":
            red = image[:,:,0]
            green = image[:,:,1]
            blue = image[:,:,2]
        r = red.astype('float64') # Make it float so that the sum can exceed value of 255 which is the max 8-bit value can be handled
        g = green.astype('float64')
        b = blue.astype('float64')
        sum = r + g + b
        red_nor = np.true_divide(red,sum)
        return red_nor
    
    def normalizeHsv(self, image):
        # Normalize HSV color
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.color == "hue":
            h = image_hsv[:,:,0]
            s = image_hsv[:,:,1]
            v = image_hsv[:,:,2]
            hue = h.astype('float64') # Make it float so that the sum can exceed value of 255 which is the max 8-bit value can be handled
            sat = s.astype('float64')
            val = v.astype('float64')
            hue_nor = np.true_divide(hue,179)
        elif self.color == "saturation":
            h = image_hsv[:,:,1]
            s = image_hsv[:,:,0]
            v = image_hsv[:,:,2]
            hue = h.astype('float64') # Make it float so that the sum can exceed value of 255 which is the max 8-bit value can be handled
            sat = s.astype('float64')
            val = v.astype('float64')
            hue_nor = np.true_divide(sat,255)
        elif self.color == "value":
            h = image_hsv[:,:,2]
            s = image_hsv[:,:,1]
            v = image_hsv[:,:,0]
            hue = h.astype('float64') # Make it float so that the sum can exceed value of 255 which is the max 8-bit value can be handled
            sat = s.astype('float64')
            val = v.astype('float64')
            hue_nor = np.true_divide(val,255)
        return hue_nor
    
    def shearStress(self, hue_array):
        # Computes Shear stress
        shear_stress = np.empty((hue_array.shape[0],hue_array.shape[1]), dtype = np.float64)
        for i in range(hue_array.shape[0]):
            for j in range(hue_array.shape[1]):
                shear_stress[i,j] = 17538*(hue_array[i,j])**4 - 13033*(hue_array[i,j])**3 + 2943.6*(hue_array[i,j])**2 + 259.89*(hue_array[i,j]) + 3.9548 #Fourth order polynomial obtained from TI511 LC curve
        return shear_stress
    
    def skinFriction(self, shear_array):
        # Computes Skin friction
        rho = 1.293 #kgm-3
        velocity = 297.135 #ms-1
        skin_friction = np.empty((shear_array.shape[0], shear_array.shape[1]), dtype = np.float64)
        for i in range(shear_array.shape[0]):
            for j in range(shear_array.shape[1]):
                skin_friction[i,j] = shear_array[i,j]/(0.5*rho*velocity**2)
        return skin_friction
    
    def contourDetection(self, image, normalized_color_array, interval):
        # Detects contour on the given image and returns the contour points, contour values and closed contour centroids
        all_contours = []
        all_centroids = []
        all_isovals = []
        all_endpoints=[]
        for i in np.linspace(round(np.amin(normalized_color_array),4),round(np.amax(normalized_color_array),4), interval, endpoint=True):
            contours = measure.find_contours(normalized_color_array, i)
            for cnt in contours:
                x1 = cnt[0,0]
                y1 = cnt[0,1]
                x2 = cnt[-1,0]
                y2 = cnt[-1,1]
                cnt_img = np.zeros((self.h,self.w), np.uint8)
                for l in range(cnt.shape[0]):
                    u = round(cnt[l,0])
                    v = round(cnt[l,1])
                    cnt_img[u,v] = 255 # draw the current contour from the array onto the black image for detecting its properties like area and centroid
                blank_image = np.zeros((self.h,self.w), np.uint8) # just a plain black image
                cnt_img = cv2.bitwise_or(cnt_img, blank_image, mask = None) # contour drawn on black image
                if x1 == x2 and y1 == y2: # if end points are same i.e. contour is a closed one then
                    labels = measure.label(cnt_img) # detect the region and label it
                    props = measure.regionprops(labels, cnt_img) # get region properties
                    cnt_area = props[0]['area'] #area is needed to threshold closed contours to have a minimum area
                    cnt_centroid = props[0]['centroid'] #centroid is needed so as to place the value as text in the final contour image
                    if cnt_area < self.contour_area_th:
                        pass
                    else:
                        all_contours.append(cnt)
                        all_centroids.append(cnt_centroid) #choosing the centroid of the closed contour for placement of text
                        all_endpoints.append(cnt_centroid)
                        all_isovals.append(i)
                else:
                    all_contours.append(cnt)
                    all_centroids.append((cnt[0,0],cnt[0,1])) #choosing the endpoint for placement of text
                    all_endpoints.append((cnt[0,0],cnt[0,1]))
                    all_isovals.append(i)
        return all_contours, all_isovals, all_centroids, all_endpoints
    
    def drawFigure(self, contour_list, isoval_list, centroid_list, color_normalized):
        # Draws figure as a graymap with contour lines
        fig = plt.figure(frameon=False, figsize = (self.w/72, self.h/72), dpi=72)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(color_normalized, cmap=plt.cm.gray, aspect = 'auto')
        colors_list = self.colorPalette() #colors list for contour lines
        prev_val = 100
        values = []
        line_colors = []
        i = 0
        for contours, val in zip(contour_list, isoval_list):
            if val == prev_val:
                ax.plot(contours[:, 1], contours[:, 0], color = colors_list[j], linewidth=2)
            else:
                ax.plot(contours[:, 1], contours[:, 0], color = colors_list[i], linewidth=2)
                values.append(val)
                line_colors.append(colors_list[i])
                j = i
                i = i + 1
            prev_val = val
        print(values)
        return fig
        
    def fig2data(self, figure):
        # Convert from matplotlib figure to opencv type so it can be saved easily
        figure.canvas.draw()
        buf = figure.canvas.tostring_rgb()
        ncols, nrows = figure.canvas.get_width_height()
        self.cv = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    
    def createLabels(self, value):
        # Create labels for contour values to be placed on image
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.5
        fontColor              = [0,0,0]
        lineType               = 1
        l_w = 40 #70
        l_h = 20
        white_img = 255*np.ones([l_h, l_w, 3],dtype=np.uint8)
        return cv2.putText(white_img, str(round(value,2)), (l_w//4-7,l_h//2+5), font, fontScale, fontColor, lineType)
    
    def opencvToPil(self, figure):
        # Converts opencv image to PIL
        figure_converted = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(figure_converted)
        
    def colorPalette(self):
        # Color Palette for contour lines
        if self.color == "blue":
            color_palette = ['#FF8A8A','#FF5C5C','#FF2E2E','#FF0000','#D10000','#A30000','#750000','#800000','#400000','#000000']
        elif self.color == "green":
            color_palette = ['#80c904','#73b504','#66a103','#5a8d03','#4d7902','#30642b','#285424','#183215','#10210e','#081107']
        elif self.color == "red":
            color_palette = ['#b8cbe7','#7b9ed2','#4072bc','#2e4a9e','#273f87','#213571','#1a2a5a','#142044','#0d152d','#060a16']
        else:
            color_palette = ['#FF0000','#800000','#FFFF00','#808000','#00FF00','#008000','#00FFFF','#008080','#0000FF','#000080','#FF00FF','#800080']
        return color_palette
        
    def addText(self, isoval_list, centroid_list, endpoints_list):
        # Adding the value labels next to contour lines in the final image
        fontScale              = 0.5
        fontColor              = [0,0,0]
        lineType               = 1
        WHITE = [255,255,255]
        contour_img = cv2.copyMakeBorder(self.cv,50,50,50,50,cv2.BORDER_CONSTANT,value=WHITE) #add borders to image
        contour_img_pil = self.opencvToPil(contour_img) #convert to PIL image so value labels can be pasted easily
        contour_with_vals = contour_img_pil.copy()
        for point, cntrd, val in zip(endpoints_list, centroid_list, isoval_list): # loop to paste all values next to contours
            label_img = self.createLabels(val)
            label_img_pil = self.opencvToPil(label_img)
            if point[0] < 10:
                label_img_rot = label_img_pil.transpose(PIL.Image.ROTATE_90)
                contour_with_vals.paste(label_img_rot, (round(cntrd[1])+40,round(cntrd[0])+5)) # top portion of image
            elif point[0] > self.h-10:
                label_img_rot = label_img_pil.transpose(PIL.Image.ROTATE_90)
                contour_with_vals.paste(label_img_rot, (round(cntrd[1])+40,round(cntrd[0])+55)) # bottom portion of image
            elif point[1] < 10:
                contour_with_vals.paste(label_img_pil, (round(cntrd[1]),round(cntrd[0])+50)) # left portion of image
            elif point[1] > self.w-10:
                contour_with_vals.paste(label_img_pil, (round(cntrd[1])+50,round(cntrd[0])+50)) # right portion of image
            else:
                contour_with_vals.paste(label_img_pil, (round(cntrd[1])+50,round(cntrd[0])+50))
        self.contour = contour_with_vals
