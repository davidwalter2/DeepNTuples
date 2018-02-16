#IN WORK

#print variable distributions of deepntuples
#   itinitialize variable and bin range
#   add fill histogram


import pdb
import ROOT
import os
from array import array

import matplotlib.pyplot as plt
import numpy as np

ROOT.gROOT.SetBatch()  # don't pop up canvases
ROOT.gROOT.SetStyle('Plain')  # white background
ROOT.gStyle.SetFillStyle(0)

class histcollection:
    """ DOC """

    collection = []


    def __init__(self, name_variable, bins):
        print("initialize "+str(name_variable))
        self.name_variable = name_variable
        self.stack = ROOT.THStack("stack_"+self.name_variable, "stack of "+self.name_variable+" hists")

        self.hists = []
        self.names_hist = 'isB', 'isBB', 'isGBB', 'isLeptonicB', 'isLeptonicB_C', 'isC', 'isGCC', 'isCC', 'isUD', 'isS', 'isG', 'isUndefined'
        self.colors = 632, 800, 801, 880, 881 , 600, 601, 602, 416, 417, 840, 920

        for iname in self.names_hist:
            self.hists.append(ROOT.TH1F(iname+"_"+self.name_variable, iname+"_"+self.name_variable+" jets", len(bins)-1, bins))

        self.hist_data = ROOT.TH1F("data_"+self.name_variable, ""+self.name_variable+" jets", len(bins)-1, bins)

        for i, hist in enumerate(self.hists):
            hist.SetLineColor(self.colors[i])

        histcollection.collection.append(self)


    def make_stack(self):
        for hist in self.hists:
            self.stack.Add(hist)
        self.hist_sum = ROOT.TH1F(self.stack.GetStack().Last())
        self.hist_sum.SetLineColor(1)

    def plot_hists(self):

        leg = ROOT.TLegend(0.59, 0.49, 0.89, 0.89)
        leg.SetBorderSize(0)
        leg.SetTextFont(42)

        for i, hist in enumerate(self.hists):
            leg.AddEntry(hist, self.names_hist[i], "f")

        canvas = ROOT.TCanvas()
        self.stack.Draw("HIST nostack")
        leg.Draw()

        canvas.Print("hists_" + self.name_variable + ".png")

    def plot_sum(self):

        leg = ROOT.TLegend(0.59, 0.79, 0.89, 0.89)
        leg.SetBorderSize(0)
        leg.SetTextFont(42)
        leg.AddEntry(self.hist_sum, 'MC', 'f')
        leg.AddEntry(self.hist_data,'data', 'lep')

        canvas = ROOT.TCanvas()
        self.hist_sum.Draw("HIST")
        self.hist_data.Draw("PE same")
        ROOT.gPad.RedrawAxis()  # draw axis in foreground
        leg.Draw()

        canvas.Print(self.name_variable + ".png")





class stackCollection:
    """ DOC """


    def __init__(self):

        #ADD variable initialization you want to plot
        self.s_jet_pt = histcollection('jet_pt', array('d', range(0,320,20)))
        self.s_jet_eta = histcollection('jet_eta', array('d', [-2.4,-2.2,-2.,-1.8,-1.6,-1.4,-1.2,-1.,-.8,-.6,-.4,-.2,0.,.2,.4,.6,.8,1.,1.2,1.4,1.6,1.8,2.,2.2,2.4]))
        self.s_jet_nCpfcand = histcollection('nCpfcand', array('d', range(0,26,1)))
        self.s_jet_nNpfcand = histcollection('nNpfcand', array('d', range(0,26,1)))
        self.s_jet_nsv = histcollection('nsv', array('d', range(0,26,1)))
        self.s_jet_npv = histcollection('npv', array('d', range(0,26,1)))
        # self.s_jet_TagVarCSV_trackSumJetEtRatio = histcollection('TagVarCSV_trackSumJetEtRatio', array('d', range(0, 26, 1)))
        # self.s_jet_TagVarCSV_trackSumJetDeltaR = histcollection('TagVarCSV_trackSumJetDeltaR', array('d', range(0, 26, 1)))
        # self.s_jet_TagVarCSV_vertexCategory = histcollection('TagVarCSV_vertexCategory', array('d', range(0, 26, 1)))
        # self.s_jet_TagVarCSV_trackSip2dValAboveCharm = histcollection('TagVarCSV_trackSip2dValAboveCharm', array('d', range(0, 26, 1)))
        # self.s_jet_TagVarCSV_trackSip2dSigAboveCharm = histcollection('TagVarCSV_trackSip2dSigAboveCharm', array('d', range(0,26,1)))
        # self.s_jet_TagVarCSV_trackSip3dValAboveCharm = histcollection('TagVarCSV_trackSip3dValAboveCharm', array('d', range(0, 26, 1)))
        # self.s_jet_TagVarCSV_trackSip3dSigAboveCharm = histcollection('TagVarCSV_trackSip3dSigAboveCharm', array('d', range(0, 26, 1)))
        # self.s_jet_TagVarCSV_jetNSelectedTracks = histcollection('TagVarCSV_jetNSelectedTracks', array('d', range(0, 26, 1)))
        # self.s_jet_TagVarCSV_jetNTracksEtaRel = histcollection('TagVarCSV_jetNTracksEtaRel', array('d', range(0, 26, 1)))

        self.s_leading_Cpfcan_pt = histcollection('leading_Cpfcan_pt', array('d', range(0,105,5)))
        self.s_leading_Npfcan_pt = histcollection('leading_Npfcan_pt', array('d', range(0,105,5)))
        self.s_leading_sv_pt = histcollection('leading_sv_pt', array('d', range(0,105,5)))

        # self.s_Cpfcan_BtagPf_trackEtaRel = histcollection('Cpfcan_BtagPf_trackEtaRel', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_BtagPf_trackPtRel = histcollection('Cpfcan_BtagPf_trackPtRel', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_BtagPf_trackPPar = histcollection('Cpfcan_BtagPf_trackPPar', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_BtagPf_trackDeltaR = histcollection('Cpfcan_BtagPf_trackDeltaR', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_BtagPf_trackPParRatio = histcollection('Cpfcan_BtagPf_trackPParRatio', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_BtagPf_trackSip2dVal = histcollection('Cpfcan_BtagPf_trackSip2dVal', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_BtagPf_trackSip2dSig = histcollection('Cpfcan_BtagPf_trackSip2dSig', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_BtagPf_trackSip3dVal = histcollection('Cpfcan_BtagPf_trackSip3dVal', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_BtagPf_trackSip3dSig = histcollection('Cpfcan_BtagPf_trackSip3dSig', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_BtagPf_trackJetDistVal = histcollection('Cpfcan_BtagPf_trackJetDistVal', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_ptrel = histcollection('Cpfcan_ptrel', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_drminsv = histcollection('Cpfcan_drminsv', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_VTX_ass = histcollection('Cpfcan_VTX_ass', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_puppiw = histcollection('Cpfcan_puppiw', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_chi2 = histcollection('Cpfcan_chi2', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        # self.s_Cpfcan_quality = histcollection('Cpfcan_quality', array('d', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))


    def fill(cls,filename):
        print 'fill hist from '+filename
        rootfile = ROOT.TFile(filename)
        tree = rootfile.Get("deepntuplizer/tree")

        for i,entry in enumerate(tree):

            weight = entry.event_weight

            isB = entry.isB
            isBB = entry.isBB
            isGBB = entry.isGBB
            isLeptonicB = entry.isLeptonicB
            isLeptonicB_C = entry.isLeptonicB_C
            isC = entry.isC
            isGCC = entry.isGCC
            isCC = entry.isCC
            isUD = entry.isUD
            isS = entry.isS
            isG = entry.isG
            isUndefined = entry.isUndefined

            flavourList = isB, isBB, isGBB, isLeptonicB, isLeptonicB_C, isC, isGCC, isCC, isUD, isS, isG, isUndefined
            for istack in histcollection.collection:
                for j, hist in enumerate(istack.hists):
                    if flavourList[j] == 1:
                        #ADD variable to fill in mc hist
                        hist.Fill(entry.jet_pt,weight)
                        hist.Fill(entry.jet_eta,weight)
                        hist.Fill(entry.nCpfcand,weight)
                        hist.Fill(entry.nNpfcand,weight)
                        hist.Fill(entry.nsv,weight)
                        hist.Fill(entry.npv,weight)
                        # hist.Fill(entry.TagVarCSV_trackSumJetEtRatio, weight)
                        # hist.Fill(entry.TagVarCSV_trackSumJetDeltaR, weight)
                        # hist.Fill(entry.TagVarCSV_vertexCategory, weight)
                        # hist.Fill(entry.TagVarCSV_trackSip2dValAboveCharm, weight)
                        # hist.Fill(entry.TagVarCSV_trackSip2dSigAboveCharm, weight)
                        # hist.Fill(entry.TagVarCSV_trackSip3dValAboveCharm, weight)
                        # hist.Fill(entry.TagVarCSV_trackSip3dSigAboveCharm, weight)
                        # hist.Fill(entry.TagVarCSV_jetNSelectedTracks, weight)
                        # hist.Fill(entry.TagVarCSV_jetNTracksEtaRel, weight)


                        if (entry.nCpfcand != 0):
                            hist.Fill(max(entry.Cpfcan_pt), weight)

                        if (entry.nNpfcand != 0):
                            hist.Fill(max(entry.Npfcan_pt), weight)

                        if (entry.nsv != 0):
                            hist.Fill(max(entry.sv_pt), weight)

                        # for i in range(0,int(entry.nCpfcand)):
                        #     hist.Fill(entry.Cpfcan_BtagPf_trackEtaRel[i], weight)
                        #     hist.Fill(entry.Cpfcan_BtagPf_trackEtaRel[i], weight)
                        #     hist.Fill(entry.Cpfcan_BtagPf_trackPtRel[i], weight)
                        #     hist.Fill(entry.Cpfcan_BtagPf_trackPPar[i], weight)
                        #     hist.Fill(entry.Cpfcan_BtagPf_trackDeltaR[i], weight)
                        #     hist.Fill(entry.Cpfcan_BtagPf_trackPParRatio[i], weight)
                        #     hist.Fill(entry.Cpfcan_BtagPf_trackSip2dVal[i], weight)
                        #     hist.Fill(entry.Cpfcan_BtagPf_trackSip2dSig[i], weight)
                        #     hist.Fill(entry.Cpfcan_BtagPf_trackSip3dVal[i], weight)
                        #     hist.Fill(entry.Cpfcan_BtagPf_trackSip3dSig[i], weight)
                        #     hist.Fill(entry.Cpfcan_BtagPf_trackJetDistVal[i], weight)
                        #     hist.Fill(entry.Cpfcan_ptrel[i], weight)
                        #     hist.Fill(entry.Cpfcan_drminsv[i], weight)
                        #     hist.Fill(entry.Cpfcan_VTX_ass[i], weight)
                        #     hist.Fill(entry.Cpfcan_puppiw[i], weight)
                        #     hist.Fill(entry.Cpfcan_chi2[i], weight)
                        #     hist.Fill(entry.Cpfcan_quality[i], weight)


            i += 1
            if i%1000 == 0:
                print(i,"entries processed")

    def fill_data(cls,filename):
        print 'fill data hist'
        rootfile = ROOT.TFile(filename)
        tree = rootfile.Get("deepntuplizer/tree")

        for i,entry in enumerate(tree):

            weight = entry.event_weight

            for istack in histcollection.collection:
                #ADD variable to fill in data hist
                istack.hist_data.Fill(entry.jet_pt,weight)
                istack.hist_data.Fill(entry.jet_eta,weight)
                istack.hist_data.Fill(entry.nCpfcand,weight)
                istack.hist_data.Fill(entry.nNpfcand,weight)
                istack.hist_data.Fill(entry.nsv,weight)
                istack.hist_data.Fill(entry.npv,weight)
                # istack.hist_data.Fill(entry.TagVarCSV_trackSumJetEtRatio, weight)
                # istack.hist_data.Fill(entry.TagVarCSV_trackSumJetDeltaR, weight)
                # istack.hist_data.Fill(entry.TagVarCSV_vertexCategory, weight)
                # istack.hist_data.Fill(entry.TagVarCSV_trackSip2dValAboveCharm, weight)
                # istack.hist_data.Fill(entry.TagVarCSV_trackSip2dSigAboveCharm, weight)
                # istack.hist_data.Fill(entry.TagVarCSV_trackSip3dValAboveCharm, weight)
                # istack.hist_data.Fill(entry.TagVarCSV_trackSip3dSigAboveCharm, weight)
                # istack.hist_data.Fill(entry.TagVarCSV_jetNSelectedTracks, weight)
                # istack.hist_data.Fill(entry.TagVarCSV_jetNTracksEtaRel, weight)


                if (entry.nCpfcand != 0):
                    istack.hist_data.Fill(max(entry.Cpfcan_pt), weight)

                if (entry.nNpfcand != 0):
                    istack.hist_data.Fill(max(entry.Npfcan_pt), weight)

                if (entry.nsv != 0):
                    istack.hist_data.Fill(max(entry.sv_pt), weight)

                # for i in range(0, int(entry.nCpfcand)):
                #     istack.hist_data.Fill(entry.Cpfcan_BtagPf_trackEtaRel[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_BtagPf_trackEtaRel[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_BtagPf_trackPtRel[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_BtagPf_trackPPar[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_BtagPf_trackDeltaR[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_BtagPf_trackPParRatio[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_BtagPf_trackSip2dVal[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_BtagPf_trackSip2dSig[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_BtagPf_trackSip3dVal[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_BtagPf_trackSip3dSig[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_BtagPf_trackJetDistVal[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_ptrel[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_drminsv[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_VTX_ass[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_puppiw[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_chi2[i], weight)
                #     istack.hist_data.Fill(entry.Cpfcan_quality[i], weight)



            i += 1
            if i%10000 == 0:
                print(i,"entries processed")

    def scale_hists_data(self,scalefactor):
        for istack in histcollection.collection:
                istack.hist_data.Scale(scalefactor)


    def make_stacks(self):
        print 'make stacks'
        for istack in histcollection.collection:
            istack.make_stack()

    def plot_hists(self):
        print 'plot stacks'
        for istack in histcollection.collection:
            istack.plot_hists()

    def plot_sums(self):
        print 'plot stacks'
        for istack in histcollection.collection:
            istack.plot_sum()

    def save_collection(self, filename):
        f1 = ROOT.TFile(filename, "RECREATE", "PlotGauss")

        for istack in histcollection.collection:
            for hist in istack.hists:
                hist.Write()
            istack.stack.Write()




col = stackCollection()
col.fill_data("muonEG_H_0.root")
col.fill("tt_1_0.root")
col.fill("tt_2_0.root")
col.fill("tt_3_0.root")
col.fill("tt_4_0.root")
col.fill("tt_5_0.root")
col.fill("dy50_1_0.root")
col.fill("dy10to50_1_0.root")
col.fill("wantit_1_0.root")
col.fill("wt_1_0.root")
col.fill("ww_1_0.root")
col.fill("wz_1_0.root")
col.fill("zz_1_0.root")
col.fill("wjets_1_0.root")

col.scale_hists_data(0.8701594758966494*35.9/8.651)

col.make_stacks()

title='variables'

directory = os.path.dirname('./plots_'+title+'/')
# make a canvas, draw, and save it
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

col.plot_hists()
col.plot_sums()
col.save_collection("collection.root")
