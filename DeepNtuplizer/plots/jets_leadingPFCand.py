#########
# Plot several variable distributions of Jets from DeepNtuplizer
#########
import pdb
import ROOT
import os

class HistStacks:
    """ DOC """
    def __init__(self,name):
        self.hists = []
        self.dataHist = []
        self.minY = 0
        self.maxY = 0
        self.legX = 0.65
        self.legY = 0.6
        self.setDefault = True
        self.name = name
        self.stack = ROOT.THStack("stack_"+self.name, "stack of "+self.name+" hists")

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

    def setLegPos(self, x,y):
        self.legX = x
        self.legY = y

    def setlabelXY(self, x = '', y = ''):
        self.labelX = x
        self.labelY = y

    def removeStatusBoxes(self):
        self.dataHist.SetStats(ROOT.kFALSE)
        for hist in self.hists:
            hist.SetStats(ROOT.kFALSE)

    def drawStack(self):

        self.removeStatusBoxes()
        self.makeStack()

        leg = ROOT.TLegend(self.legX, self.legY, self.legX + 0.15, self.legY + 0.25)
        leg.SetBorderSize(0)
        leg.SetTextFont(42)
        leg.AddEntry(self.dataHist, "data", "lep")
        leg.AddEntry(self.hists[8], "t#bar{t}", "f")
        leg.AddEntry(self.hists[7], "WJets", "f")
        leg.AddEntry(self.hists[5], "VV", "f")
        leg.AddEntry(self.hists[3], "Wt/W#bar{t}", "f")
        leg.AddEntry(self.hists[1], "DY", "f")



        self.dataHist.SetMinimum(0)
        if not self.setDefault:
            self.dataHist.SetMinimum(self.minY)
            self.dataHist.SetMaximum(self.maxY)

        self.dataHist.GetXaxis().SetTitle(self.labelX)
        self.dataHist.GetYaxis().SetTitle(self.labelY)
        self.dataHist.SetMarkerStyle(ROOT.kFullDotLarge)
        self.dataHist.SetMarkerSize(0.5)

        canvas = ROOT.TCanvas()
        self.dataHist.Draw("AXIS")
        self.stack.Draw("HIST same")
        self.dataHist.Draw("PE same")
        leg.Draw("same")
        ROOT.gPad.RedrawAxis()      #draw axis in foreground
        canvas.Print("stack_"+self.name+".png")



class Source:
    """ DOC """

    stack_Cpfcan_pt = HistStacks("Cpfcan_pt")
    stack_Npfcan_pt = HistStacks("Npfcan_pt")


    def __init__(self, name, color = ROOT.kBlack,  data=False):
        self.name = name
        self.data = data
        self.rootfile = ROOT.TFile(self.name+"_0.root")
        self.tree = self.rootfile.Get("deepntuplizer/tree")

        self.hist_Cpfcan_pt = ROOT.TH1F("Cpfcan_pt"+self.name, "leading charged particle flow candidates of "+self.name, 40, 0, 100)
        self.hist_Npfcan_pt = ROOT.TH1F("Npfcan_pt"+self.name, "leading neutral particle flow candidates of "+self.name, 40, 0, 100)

        self.hists = self.hist_Cpfcan_pt, self.hist_Npfcan_pt

        self.setHistColor(color)

        self.fillHists()

        if data:
            Source.stack_Cpfcan_pt.addDataHist(self.hist_Cpfcan_pt)
            Source.stack_Npfcan_pt.addDataHist(self.hist_Npfcan_pt)

        else:
            self.scaleHists(0.8180639286773081)

            Source.stack_Cpfcan_pt.addHist(self.hist_Cpfcan_pt)
            Source.stack_Npfcan_pt.addHist(self.hist_Npfcan_pt)





    def fillHists(self):
        print("fill hists of ",self.name," sample")
        i = 0

        for entry in self.tree:

            weight = entry.jet_weight

            Cpfcan_pt = entry.Cpfcan_pt
            Npfcan_pt = entry.Npfcan_pt
            if(len(Cpfcan_pt) != 0):
                self.hist_Cpfcan_pt.Fill(max(Cpfcan_pt), weight)
            if(len(Npfcan_pt) != 0):
                self.hist_Npfcan_pt.Fill(max(Npfcan_pt), weight)

            i += 1
            if i%10000 == 0:
                print(i,"entries processed")
                #break



    def setHistColor(self,color=ROOT.kBlack):
        for hist in self.hists:
            hist.SetFillColor(color)

    def scaleHists(self,factor):
        for hist in self.hists:
            hist.Scale(factor)

    def draw_all(self):
        i = 0
        for hist in self.hists:
            c1 = ROOT.TCanvas()
            if self.data:
                hist.Draw("PE")
            else:
                hist.Draw("HIST")
            c1.Print(self.name + "_Jet_"+str(i)+".png")
            i += 1

    @classmethod
    def draw_stacks(cls):

        cls.stack_Cpfcan_pt.setMinMax(0,90000)
        cls.stack_Npfcan_pt.setMinMax(0,140000)

        #cls.stack_eta.setLegPos(0.15,0.6)

        cls.stack_Cpfcan_pt.dataHist.GetXaxis().SetNdivisions(4)
        cls.stack_Npfcan_pt.dataHist.GetXaxis().SetNdivisions(4)


        cls.stack_Cpfcan_pt.setlabelXY('pt of leading charged pf cand.','events')
        cls.stack_Npfcan_pt.setlabelXY('pt of leading neutral pf cand.','events')


        cls.stack_Cpfcan_pt.drawStack()
        cls.stack_Npfcan_pt.drawStack()



ROOT.gROOT.SetBatch()  # don't pop up canvases
ROOT.gROOT.SetStyle('Plain')  # white background
ROOT.gStyle.SetFillStyle(0)  # TPave objects (e.g. legend) are transparent
ROOT.gStyle.SetOptTitle(0)  # no title
ROOT.TGaxis.SetMaxDigits(4)  # Force scientific notation for numbers with more than 4 digits
ROOT.gStyle.SetTextFont(42)
ROOT.gStyle.SetTitleFont(42, "t")
ROOT.gStyle.SetTitleFont(42, "xyz")
ROOT.gStyle.SetLabelFont(42, "xyz")


        
data = Source("data", data=True)
dy50 = Source("dy50",ROOT.kBlue)      #632
dy10to50 = Source("dy10to50",ROOT.kBlue)
wantit = Source("wantit",ROOT.kMagenta)                #797
wt = Source("wt",ROOT.kMagenta)
ww = Source("ww",ROOT.kYellow)      #800
wz = Source("wz",ROOT.kYellow)
zz = Source("zz",ROOT.kYellow)
wjets = Source("wjets",ROOT.kGreen) #
tt = Source("tt",ROOT.kRed)

print("make directory and save plots")
directory = os.path.dirname('./plots_jets_pfcands/')
# make a canvas, draw, and save it
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)


Source.draw_stacks()
