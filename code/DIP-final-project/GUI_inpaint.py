from copy import deepcopy
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import numpy as np
import cv2
import time
import skimage

from main import main
from pconv import inpaint_pconv
from criminisi import inpaint_criminisi

class app(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.w = self.h = 512
        self.inpaint_cnt = 0

        self.title('inpainter')

        self.canvas1 = Canvas(self, width=self.w,height=self.h, bd=0, bg='white')
        self.canvas1.grid(row=1, column=0)
        self.canvas2 = Canvas(self, width=self.w,height=self.h, bd=0, bg='white')
        self.canvas2.grid(row=1, column=2)
        self.mask = np.zeros((self.w, self.h), dtype=np.uint8)
        self.mask = cv2.merge([self.mask, self.mask, self.mask])

        self.bind("<B1-Motion>", self.paint)

        self.ori_mode = True
        self.exchange_button = Button(self, text='original image', command=self.exchangeImg)
        self.exchange_button.grid(row=0, column=0)

        self.output_mode = 0
        self.exchange_button2 = Button(self, text='output image', command=self.exchangeImg2)
        self.exchange_button2.grid(row=0, column=2)

        self.ori_button = Button(self, text='open a image', command=self.loadOri)
        self.ori_button.grid(row=3, column=0)

        self.mask_button = Button(self, text='load mask', command=self.loadMask)
        self.mask_button.grid(row=3, column=1)

        self.clear_button = Button(self, text='clear mask', command=self.clearMask)
        self.clear_button.grid(row=2, column=1)

        self.run_button = Button(self, text='RUN', command=self.run_model)
        self.run_button.grid(row=3, column=2)

        self.log_text = Text(self, width=50, height=5)
        self.log_text.grid(row=4, columnspan=4)
        self.log_text.insert('insert', ' ')

        self.algorithm = [['Edge-Connect'],
                          ['Partial-Conv'],
                          ['TELEA'],
                          ['Navier-Stokes'],
                          ['Criminisi']]

        self.algorithmList = Listbox(self, width=15, height=len(self.algorithm), selectmode=EXTENDED)
        for idx in range(len(self.algorithm)):
            self.algorithmList.insert(idx + 1, self.algorithm[idx])
        self.algorithmList.grid(row=1, column=1)

    def paint(self, event):
        cv2.circle(self.mask, (event.x, event.y), 6, (255, 255, 255), -1)
        self.ori_mode = True
        self.exchangeImg()

    def exchangeImg(self):
        self.ori_mode ^= 1
        if (self.ori_mode):
            imgfile = ImageTk.PhotoImage(self.ori_img)
            self.exchange_button['text'] = 'original image'
        else:
            self.masked_Img = cv2.cvtColor(np.asarray(deepcopy(self.ori_img)), cv2.COLOR_RGB2BGR)
            self.masked_Img[self.mask > 0] = 255
            self.masked_Img = Image.fromarray(cv2.cvtColor(self.masked_Img, cv2.COLOR_BGR2RGB))
            imgfile = ImageTk.PhotoImage(self.masked_Img)
            self.exchange_button['text'] = 'masked image'

        self.canvas1.image = imgfile  # <--- keep reference of your image
        self.canvas1.create_image(2, 2, anchor='nw', image=imgfile)
        self.canvas1.old_coords = None

    def exchangeImg2(self):
        self.output_mode += 1
        self.output_mode %= 3
        if self.output_mode == 0:
            imgfile = ImageTk.PhotoImage(Image.open('tmp/output.png').resize((self.w, self.h)))
            self.exchange_button2['text'] = 'output image'
        elif self.output_mode == 1:
            self.masked_Img = cv2.cvtColor(np.asarray(deepcopy(self.ori_img)), cv2.COLOR_RGB2BGR)
            self.masked_Img[self.mask > 0] = 255
            self.masked_Img = Image.fromarray(cv2.cvtColor(self.masked_Img, cv2.COLOR_BGR2RGB))
            imgfile = ImageTk.PhotoImage(self.masked_Img)
            self.exchange_button2['text'] = 'masked image'
        else:
            imgfile = ImageTk.PhotoImage(self.ori_img)
            self.exchange_button2['text'] = 'original image'

        self.canvas2.image = imgfile  # <--- keep reference of your image
        self.canvas2.create_image(2, 2, anchor='nw', image=imgfile)

    #the class to show the original image
    def loadOri(self):
        File = askopenfilename(title='Open Image')

        self.ori_img = Image.open(File).resize((self.w, self.h))
        self.ori_img.save('tmp/ori.png')
        imgfile = ImageTk.PhotoImage(self.ori_img)

        self.canvas1.image = imgfile  # <--- keep reference of your image
        self.canvas1.create_image(2, 2, anchor='nw', image=imgfile)
        self.canvas1.old_coords = None

    #the class to show the inpainted image   
    def loadMask(self):
        File = askopenfilename(title='Open Mask')
        self.mask = 255 * np.array(np.array(Image.open(File).resize((self.w, self.h)).convert('L')) > 0, dtype=np.uint8)
        self.mask = cv2.merge([self.mask, self.mask, self.mask])

        self.ori_mode = True
        self.exchangeImg()

    def clearMask(self):
        self.mask = np.zeros((self.w, self.h), dtype=np.uint8)
        self.mask = cv2.merge([self.mask, self.mask, self.mask])

        self.ori_mode = True
        self.exchangeImg()

    def Metrics(self):
        im1 = skimage.io.imread('tmp/ori.png')
        im2 = skimage.io.imread('tmp/output.png')
        psnr = skimage.measure.compare_psnr(im1, im2, 255)
        ssim = skimage.measure.compare_ssim(im1, im2, data_range=255, multichannel=True)
        return [psnr, ssim]

    def run_model(self):
        cv2.imwrite('tmp/mask.png', self.mask)

        model = self.algorithm[self.algorithmList.curselection()[0]]

        inputFile = 'tmp/input.png'
        maskFile = 'tmp/mask.png'
        outputFile = 'tmp/output.png'

        self.masked_Img.save(inputFile)
        cv2.imwrite(maskFile, self.mask)

        start = time.clock()

        if model == ['Edge-Connect']:
            self.masked_Img.save('tmp/input/input.png')
            cv2.imwrite('tmp/mask/input.png', self.mask)
            main(mode=2)
            Image.open('tmp/output/input.png').save(outputFile)
        elif model == ['Partial-Conv']:
            inpaint_pconv(self.h, self.w)
        elif model == ['TELEA']:
            cv2.imwrite(outputFile, cv2.inpaint(cv2.imread(inputFile), cv2.split(cv2.imread(maskFile))[0], 6, flags=cv2.INPAINT_TELEA))
        elif model == ['Navier-Stokes']:
            cv2.imwrite(outputFile, cv2.inpaint(cv2.imread(inputFile), cv2.split(cv2.imread(maskFile))[0], 6, flags=cv2.INPAINT_NS))
        elif model == ['Criminisi']:
            #self.tmp_masked_Img = cv2.resize(cv2.cvtColor(np.asarray(deepcopy(self.ori_img)), cv2.COLOR_RGB2BGR), (256, 256))
            #self.tmp_mask = 255 * (cv2.resize(self.mask, (256, 256)) > 0)
            #self.tmp_masked_Img[cv2.split(self.tmp_mask)[0] > 0] = 255
            #self.tmp_masked_Img = Image.fromarray(cv2.cvtColor(self.tmp_masked_Img, cv2.COLOR_BGR2RGB))
            #self.tmp_masked_Img.save(inputFile)
            #cv2.imwrite(maskFile, self.tmp_mask)
            inpaint_criminisi()

        elapsed = (time.clock() - start)

        imgfile = ImageTk.PhotoImage(Image.open('tmp/output.png').resize((self.w, self.h)))
        self.canvas2.image = imgfile  # <--- keep reference of your image
        self.canvas2.create_image(2, 2, anchor='nw', image=imgfile)
        self.inpaint_cnt += 1
        self.output_mode = 2
        self.exchangeImg2()

        metrics = self.Metrics()
        log = '         ----- %dth inpaint finished -----\n' % self.inpaint_cnt
        log += '    Algorithm : ' + model[0] + '\n'
        log += '    Time used : %fs\n' % elapsed
        log += '    psnr : %fdB, ssim : %f\n' % (metrics[0], metrics[1])
        log += '    total pixel to be inpainted : %d(%f%%) \n' % (np.sum(self.mask > 0) / 3, np.sum(self.mask > 0) / 3 / self.w / self.h * 100)
        print(log)
        self.log_text.delete(1.0, 'end')
        self.log_text.insert('insert', log)



root = app()
root.mainloop()

'''256*256 mask 32.45%

    Algorithm : Edge-Connect
    Time used : 8.004961s
    psnr : 15.255895dB, ssim : 0.687687

    Algorithm : Pconv
    ----
    
    Algorithm : TELEA
    Time used : 0.131862s
    psnr : 15.213468dB, ssim : 0.686095

    Algorithm : Navier-Stokes
    Time used : 0.128929s
    psnr : 15.165652dB, ssim : 0.689933

    
    

'''