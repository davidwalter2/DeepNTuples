#########
#
# Make plots about the jet composition (isB, isBB, isC, ...) in the sample of jet-deepntuples
# tested with CMSSW 8_4_0
#########
import pdb
import ROOT
import os
from array import array

import matplotlib.pyplot as plt
import numpy as np

class HistStacks:
    """ DOC """

    collection = []

    def __init__(self,name):
        self.hists = []
        self.dataHist = []
        self.minY = 0
        self.maxY = 0
        self.legX = 0.59
        self.legY = 0.45
        self.setDefault = True
        self.name = name
        self.stack = ROOT.THStack("stack_"+self.name,"")

        self.jetSum = 0.

        self.labelX = ''
        self.labelY = ''

        HistStacks.collection.append(self)

    def addHist(self, hist):
        self.hists.append(hist)

    def addDataHist(self, hist):
        self.dataHist = hist

    def makeStack(self):
        for hist in self.hists:
            self.stack.Add(hist)

    def setMinMax(self, min, max):
        self.minY = min
        self.maxY = max
        self.setDefault = False

    def setlabelXY(self, x = '', y = ''):
        self.labelX = x
        self.labelY = y

    def setLegPos(self, x,y):
        self.legX = x
        self.legY = y

    def removeStatusBoxes(self):
        if self.dataHist != []:
            self.dataHist.SetStats(ROOT.kFALSE)
        for hist in self.hists:
            hist.SetStats(ROOT.kFALSE)

    def drawStack(self):

        self.removeStatusBoxes()
        self.makeStack()

        leg = ROOT.TLegend(self.legX, self.legY, self.legX + 0.15, self.legY + 0.25)
        leg.SetBorderSize(0)
        leg.SetTextFont(42)
        #leg.AddEntry(self.dataHist, "data", "lep")
        leg.AddEntry(self.hists[8], "t#bar{t}", "f")
        leg.AddEntry(self.hists[7], "WJets", "f")
        leg.AddEntry(self.hists[5], "VV", "f")
        leg.AddEntry(self.hists[3], "Wt/W#bar{t}", "f")
        leg.AddEntry(self.hists[1], "DY", "f")

        if not self.setDefault:
            self.dataHist.SetMinimum(self.minY)
            self.dataHist.SetMaximum(self.maxY)

        canvas = ROOT.TCanvas()
        self.stack.Draw("HIST")
        self.stack.GetXaxis().SetTitle(self.labelX)
        self.stack.GetYaxis().SetTitle(self.labelY)

        self.stack.GetXaxis().SetNdivisions(2)

        leg.Draw("same")
        ROOT.gPad.RedrawAxis()      #draw axis in foreground
        canvas.Print("stack_"+self.name+".png")

    @classmethod
    def drawStacks(cls):
        for stack in cls.collection:
            stack.drawStack()
            print("sum of "+str(stack.name)+" jets = ", ROOT.TH1F(stack.stack.GetStack().Last()).GetBinContent(2), " of ",ROOT.TH1F(stack.stack.GetStack().Last()).Integral())

    @classmethod
    def drawSummaryPie(cls):

        frequencies = []
        names = []

        for stack in cls.collection:
            stack.jetSum = ROOT.TH1F(stack.stack.GetStack().Last()).GetBinContent(2)
            frequencies.append(stack.jetSum)
            names.append(stack.name)
        canvas = ROOT.TCanvas()
        #pdb.set_trace()
        pieChart = ROOT.TPie("pie4","Pie with verbose labels", 13, array('d',frequencies),array('',names))
        pieChart.Draw("rsc")
        canvas.Print("summary_pie.png")

    @classmethod
    def drawSummaryChart(cls):
        frequencies = []
        names = []

        for stack in cls.collection:
            stack.jetSum = ROOT.TH1F(stack.stack.GetStack().Last()).GetBinContent(2)
            frequencies.append(stack.jetSum)
            names.append(stack.name)


        fig, ax = plt.subplots()
        ind = np.arange(1, len(frequencies)+1)

        plt.gcf().subplots_adjust(bottom=0.25)
        plt.bar(ind, frequencies, align='center')

        ax.set_xticks(ind)
        ax.set_xticklabels(names, rotation=90)
        ax.set_ylim([0, 300000])
        ax.set_ylabel('weighted frequency')
        ax.set_title('GoodJets in MC')

        plt.savefig("summaryChart.png")

    @classmethod
    def drawSummaryChart_deepFlavour(cls):
        frequencies = []
        names = []
        sumTotal = 0
        for stack in cls.collection:
            stack.jetSum = ROOT.TH1F(stack.stack.GetStack().Last()).GetBinContent(2)
            sumTotal += stack.jetSum

        names.append('isB')
        jetSumB = ROOT.TH1F(cls.collection[0].stack.GetStack().Last()).GetBinContent(2)
        frequencies.append(float(jetSumB)/sumTotal)

        names.append('isBB')
        jetSumBB = ROOT.TH1F(cls.collection[1].stack.GetStack().Last()).GetBinContent(2)
        jetSumBB += ROOT.TH1F(cls.collection[2].stack.GetStack().Last()).GetBinContent(2)
        frequencies.append(float(jetSumBB)/sumTotal)

        names.append('isLeptB')
        jetSumLeptB = ROOT.TH1F(cls.collection[3].stack.GetStack().Last()).GetBinContent(2)
        jetSumLeptB += ROOT.TH1F(cls.collection[4].stack.GetStack().Last()).GetBinContent(2)
        frequencies.append(float(jetSumLeptB)/sumTotal)

        names.append('isC')
        jetSumC = ROOT.TH1F(cls.collection[5].stack.GetStack().Last()).GetBinContent(2)
        jetSumC += ROOT.TH1F(cls.collection[6].stack.GetStack().Last()).GetBinContent(2)
        jetSumC += ROOT.TH1F(cls.collection[7].stack.GetStack().Last()).GetBinContent(2)
        frequencies.append(float(jetSumC)/sumTotal)

        names.append('isUDS')
        jetSumLeptUDS = ROOT.TH1F(cls.collection[8].stack.GetStack().Last()).GetBinContent(2)
        jetSumLeptUDS += ROOT.TH1F(cls.collection[9].stack.GetStack().Last()).GetBinContent(2)
        frequencies.append(float(jetSumLeptUDS)/sumTotal)

        names.append('isG')
        jetSumG = ROOT.TH1F(cls.collection[10].stack.GetStack().Last()).GetBinContent(2)
        frequencies.append(float(jetSumG)/sumTotal)

        fig, ax = plt.subplots()
        ind = np.arange(1, len(frequencies)+1)

        plt.gcf().subplots_adjust(bottom=0.20)
        plt.bar(ind, frequencies, align='center')

        ax.set_xticks(ind)
        ax.set_xticklabels(names, rotation=90)
        ax.set_ylim([0, 0.5])
        ax.set_ylabel('relative weighted frequency')
        ax.set_title('GoodJets in MC - deepFlavour labels')

        plt.savefig("summaryChart_deepFlavour.png")



class Source:
    """ DOC """
    stack_isB = HistStacks("isB")
    stack_isBB = HistStacks("isBB")
    stack_isGBB = HistStacks("isGBB")
    stack_isLeptonicB = HistStacks("isLeptonicB")
    stack_isLeptonicB_C = HistStacks("isLeptonicB_C")
    stack_isC = HistStacks("isC")
    stack_isGCC = HistStacks("isGCC")
    stack_isCC = HistStacks("isCC")
    stack_isUD = HistStacks("isUD")
    stack_isS = HistStacks("isS")
    stack_isG = HistStacks("isG")
    stack_isUndefined = HistStacks("isUndefined")




    def __init__(self, name, color = ROOT.kBlack, data=False):
        print("initialize "+str(name))
        self.name = name
        self.data = data
        self.hists = []
        self.rootfile = ROOT.TFile(self.name+"_0.root")
        self.tree = self.rootfile.Get("deepntuplizer/tree")

        self.hist_isB = ROOT.TH1F("isB_"+self.name, "labels of "+self.name+" jets", 2, -0.5, 1.5)
        self.hist_isBB = ROOT.TH1F("isBB_"+self.name, "labels of "+self.name+" jets", 2, -0.5, 1.5)
        self.hist_isGBB = ROOT.TH1F("isGBB_"+self.name, "labels of "+self.name+" jets", 2, -0.5, 1.5)
        self.hist_isLeptonicB = ROOT.TH1F("isLeptonicB_"+self.name, "labels of "+self.name+" jets", 2, -0.5, 1.5)
        self.hist_isLeptonicB_C = ROOT.TH1F("isLeptonicB_C_"+self.name, "labels of "+self.name+" jets", 2, -0.5, 1.5)
        self.hist_isC = ROOT.TH1F("isC_"+self.name, "labels of "+self.name+" jets", 2, -0.5, 1.5)
        self.hist_isGCC = ROOT.TH1F("isGCC_"+self.name, "labels of "+self.name+" jets", 2, -0.5, 1.5)
        self.hist_isCC = ROOT.TH1F("isCC_"+self.name, "labels of "+self.name+" jets", 2, -0.5, 1.5)
        self.hist_isUD = ROOT.TH1F("isUD_"+self.name, "labels of "+self.name+" jets", 2, -0.5, 1.5)
        self.hist_isS = ROOT.TH1F("isS_"+self.name, "labels of "+self.name+" jets", 2, -0.5, 1.5)
        self.hist_isG = ROOT.TH1F("isG_"+self.name, "labels of "+self.name+" jets", 2, -0.5, 1.5)
        self.hist_isUndefined = ROOT.TH1F("isUndefined_"+self.name, "labels of "+self.name+" jets", 2, -0.5, 1.5)


        self.tree.Draw("isB>>isB_"+self.name, "event_weight")
        self.tree.Draw("isBB>>isBB_"+self.name, "event_weight")
        self.tree.Draw("isGBB>>isGBB_"+self.name, "event_weight")
        self.tree.Draw("isLeptonicB>>isLeptonicB_"+self.name, "event_weight")
        self.tree.Draw("isLeptonicB_C>>isLeptonicB_C_"+self.name, "event_weight")
        self.tree.Draw("isC>>isC_"+self.name, "event_weight")
        self.tree.Draw("isGCC>>isGCC_"+self.name, "event_weight")
        self.tree.Draw("isCC>>isCC_"+self.name, "event_weight")
        self.tree.Draw("isUD>>isUD_"+self.name, "event_weight")
        self.tree.Draw("isS>>isS_"+self.name, "event_weight")
        self.tree.Draw("isG>>isG_"+self.name, "event_weight")
        self.tree.Draw("isUndefined>>isUndefined_"+self.name, "event_weight")




        self.hists = self.hist_isB, self.hist_isBB, self.hist_isGBB,self.hist_isLeptonicB,self.hist_isLeptonicB_C,\
                     self.hist_isC,self.hist_isGCC,self.hist_isCC,self.hist_isUD,self.hist_isS,self.hist_isG,\
                     self.hist_isUndefined


        self.setHistColor(color)


        Source.stack_isB.addHist(self.hist_isB)
        Source.stack_isBB.addHist(self.hist_isBB)
        Source.stack_isGBB.addHist(self.hist_isGBB)
        Source.stack_isLeptonicB.addHist(self.hist_isGBB)
        Source.stack_isLeptonicB_C.addHist(self.hist_isLeptonicB_C)
        Source.stack_isC.addHist(self.hist_isC)
        Source.stack_isGCC.addHist(self.hist_isGCC)
        Source.stack_isCC.addHist(self.hist_isCC)
        Source.stack_isUD.addHist(self.hist_isUD)
        Source.stack_isS.addHist(self.hist_isS)
        Source.stack_isG.addHist(self.hist_isG)
        Source.stack_isUndefined.addHist(self.hist_isUndefined)



    def setHistColor(self,color=ROOT.kBlack):
        for hist in self.hists:
            hist.SetFillColor(color)

    def scaleHists(self,factor):
        for hist in self.hists:
            hist.Scale(factor)

    def normHists(self):
        for hist in self.hists:
            hist.Scale(1./hist.Integral())

    def draw_all(self):
        i = 0
        for hist in self.hists:
            c1 = ROOT.TCanvas()
            hist.SetMinimum(0)
            if self.data:
                hist.Draw("PE")
            else:
                hist.Draw("HIST")
            c1.Print(self.name + "_Jet_"+str(i)+".png")
            i += 1


    @classmethod
    def draw_stacks(cls):

        cls.stack_isB.removeStatusBoxes()

        cls.stack_isB.setLegPos(0.65,0.625)

        cls.stack_isB.setlabelXY('isB','events')

        cls.stack_isB.drawStack()






ROOT.gROOT.SetBatch()           # don't pop up canvases
ROOT.gROOT.SetStyle('Plain')    # white background
#ROOT.gStyle.SetFillStyle(0)     # TPave objects (e.g. legend) are transparent

ROOT.gStyle.SetTextFont(42)
ROOT.gStyle.SetTitleFont(42, "t")
ROOT.gStyle.SetTitleFont(42, "xyz")
ROOT.gStyle.SetLabelFont(42, "xyz")


dy50 = Source("dy50_1",ROOT.kBlue)            #632
dy10to50 = Source("dy10to50_1",ROOT.kBlue)
wantit = Source("wantit_1",ROOT.kMagenta)     #797
wt = Source("wt_1",ROOT.kMagenta)
ww = Source("ww_1",ROOT.kYellow)              #800
wz = Source("wz_1",ROOT.kYellow)
zz = Source("zz_1",ROOT.kYellow)
wjets = Source("wjets_1",ROOT.kGreen)         #
tt1 = Source("tt_1",ROOT.kRed)
tt2 = Source("tt_2",ROOT.kRed)
tt3 = Source("tt_3",ROOT.kRed)
tt4 = Source("tt_4",ROOT.kRed)
tt5 = Source("tt_5",ROOT.kRed)



print("make directory and save plots")
directory = os.path.dirname('./plots_labels/')
# make a canvas, draw, and save it
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

#data.draw_all()
#tt.draw_all()
#dy50.draw_all()
#dy10to50.draw_all()
#wantit.draw_all()
#wt.draw_all()
#ww.draw_all()
#wz.draw_all()
#zz.draw_all()
#wjets.draw_all()

#pdb.set_trace()


HistStacks.drawStacks()
HistStacks.drawSummaryChart()
HistStacks.drawSummaryChart_deepFlavour()


Source.draw_stacks()


