#########
# Draws the label composition of some Ntuplizer root file
#########
import pdb
import ROOT
import os
from array import array





class HistStacks:
    """ DOC """
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

        self.labelX = ''
        self.labelY = ''

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
        #leg.AddEntry(self.hists[8], "t#bar{t}", "f")
        #leg.AddEntry(self.hists[7], "WJets", "f")
        #leg.AddEntry(self.hists[5], "VV", "f")
        #leg.AddEntry(self.hists[3], "Wt/W#bar{t}", "f")
        #leg.AddEntry(self.hists[1], "DY", "f")

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


class Source:
    """ DOC """
    stack_prob_isB = HistStacks("prob_isB")

    def __init__(self, name, color = ROOT.kBlack, data=False):
        self.name = name
        self.data = data
        self.hists = []
        self.rootfile = ROOT.TFile(self.name+"_0_predict.root")
        self.tree = self.rootfile.Get("tree")

        self.hist_prob_isB = ROOT.TH1F("prob_isB_"+self.name, "predictions of "+self.name+" jets", 20, 0, 1)
        self.hist_prob_isBB = ROOT.TH1F("prob_isBB_"+self.name, "predictions of "+self.name+" jets", 20, 0, 1)


        self.tree.Draw("prob_isB>>prob_isB_"+self.name) #, "jet_weight")
        self.tree.Draw("prob_isBB>>prob_isBB_"+self.name) #, "jet_weight")


        self.hists = self.hist_prob_isB, self.hist_prob_isBB




        self.setHistColor(color)


        #self.scaleHists(0.8180639286773081)


        Source.stack_prob_isB.addHist(self.hist_prob_isB)


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

        cls.stack_prob_isB.removeStatusBoxes()

        cls.stack_prob_isB.setLegPos(0.65,0.625)

        cls.stack_prob_isB.setlabelXY('isB','events')

        cls.stack_prob_isB.drawStack()






ROOT.gROOT.SetBatch()           # don't pop up canvases
ROOT.gROOT.SetStyle('Plain')    # white background
#ROOT.gStyle.SetFillStyle(0)     # TPave objects (e.g. legend) are transparent

ROOT.gStyle.SetTextFont(42)
ROOT.gStyle.SetTitleFont(42, "t")
ROOT.gStyle.SetTitleFont(42, "xyz")
ROOT.gStyle.SetLabelFont(42, "xyz")


dy50 = Source("dy50",ROOT.kBlue)            #632
dy10to50 = Source("dy10to50",ROOT.kBlue)
wantit = Source("wantit",ROOT.kMagenta)     #797
wt = Source("wt",ROOT.kMagenta)
#ww = Source("ww",ROOT.kYellow)              #800
#wz = Source("wz",ROOT.kYellow)
#zz = Source("zz",ROOT.kYellow)
wjets = Source("wjets",ROOT.kGreen)         #
tt = Source("tt",ROOT.kRed)


print("make directory and save plots")
directory = os.path.dirname('./plots_output/')
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

Source.draw_stacks()


