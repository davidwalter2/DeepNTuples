import pdb
import ROOT
import os
import numpy as np

import FWCore.ParameterSet.Config as cms

from DataFormats.FWLite import Events, Handle
from argparse import ArgumentParser

ROOT.gROOT.SetBatch()  # don't pop up canvases
ROOT.gROOT.SetStyle('Plain')  # white background
ROOT.gStyle.SetFillStyle(0)

def makeFullPlots(h_data,hs_mc, name="all"):
    h_data[0].SetMinimum(0)
    h_data[0].SetMaximum(8000)
    h_data[1].SetMinimum(0)
    h_data[1].SetMaximum(160000)
    h_data[2].SetMinimum(0)
    h_data[2].SetMaximum(18500)
    h_data[3].SetMinimum(0)
    h_data[3].SetMaximum(150000)
    h_data[4].SetMinimum(0)
    h_data[4].SetMaximum(140000)


    leg = ROOT.TLegend(0.59, 0.49, 0.89, 0.89)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.AddEntry(h_data[0], "data", "lep")
    leg.AddEntry(hs_mc[0].GetHists()[-1], "t#bar{t}", "f")
    leg.AddEntry(hs_mc[0].GetHists()[0], "DY", "f")
    leg.AddEntry(hs_mc[0].GetHists()[4], "VV", "f")
    leg.AddEntry(hs_mc[0].GetHists()[2], "Wt/W#bar{t}", "f")
    leg.AddEntry(hs_mc[0].GetHists()[7],"WJets", "f")

    leg21 = ROOT.TLegend(0.19, 0.69, 0.39, 0.89)
    leg21.SetBorderSize(0)
    leg21.SetTextFont(42)
    leg21.AddEntry(h_data[0], "data", "lep")
    leg21.AddEntry(hs_mc[0].GetHists()[-1], "t#bar{t}", "f")
    leg21.AddEntry(hs_mc[0].GetHists()[0], "DY", "f")

    leg22 = ROOT.TLegend(0.69, 0.69, 0.89, 0.89)
    leg22.SetBorderSize(0)
    leg22.SetTextFont(42)
    leg22.AddEntry(hs_mc[0].GetHists()[4], "VV", "f")
    leg22.AddEntry(hs_mc[0].GetHists()[2], "Wt/W#bar{t}", "f")
    leg22.AddEntry(hs_mc[0].GetHists()[7],"WJets", "f")

    leg.SetFillStyle(0)     #make transparent
    leg21.SetFillStyle(0)
    leg22.SetFillStyle(0)

    h_data[0].SetTitleOffset(1.18,"y")
    for hist in h_data:
        hist.SetTitle("")
        #hist.SetTitleSize(0.05, "xyz")
        hist.SetTitleFont(42,"t")
        hist.SetTitleFont(42,"xyz")
        hist.SetLabelFont(42,"xyz")

    h_data[0].SetXTitle("leading lept. p_{T} (GeV)")
    h_data[1].SetXTitle("leading lept. eta")
    h_data[2].SetXTitle("trailing lept. p_{T} (GeV)")
    h_data[3].SetXTitle("trailing lept. eta")
    h_data[4].SetXTitle("N_{Jet}")

    h_data[0].SetYTitle("Events/GeV")
    h_data[1].SetYTitle("Events/0.25")
    h_data[2].SetYTitle("Events/GeV")
    h_data[3].SetYTitle("Events/0.25")
    h_data[4].SetYTitle("Events")

    ROOT.TGaxis.SetMaxDigits(4)   #Force scientific notation for numbers with more than 4 digits


    canvas = ROOT.TCanvas("final", "title", 800, 600)

    for i in (0,2,4):
        canvas.Clear()
        h_data[i].Draw("PE")
        hs_mc[i].Draw("HIST same")
        h_data[i].Draw("PE same")
        ROOT.gPad.RedrawAxis()      #draw axis in foreground
        leg.Draw("same")
        canvas.Print(name + "_" + str(i) + ".png")

    for i in (1,3):
        canvas.Clear()
        h_data[i].Draw("PE")
        hs_mc[i].Draw("HIST same")
        h_data[i].Draw("PE same")
        ROOT.gPad.RedrawAxis()      #draw axis in foreground
        leg21.Draw("same")
        leg22.Draw("same")
        canvas.Print(name + "_" + str(i) + ".png")

    canvasSum = ROOT.TCanvas("finalSum", "title", 800, 600)
    canvasSum.Divide(2,3)

    for i in (0,2,4):
        canvasSum.cd(i+1)
        h_data[i].Draw("PE")
        hs_mc[i].Draw("HIST same")
        h_data[i].Draw("PE same")
        ROOT.gPad.RedrawAxis()      #draw axis in foreground
        leg.Draw("same")

    for i in (1,3):
        canvasSum.cd(i+1)
        h_data[i].Draw("PE")
        hs_mc[i].Draw("HIST same")
        h_data[i].Draw("PE same")
        ROOT.gPad.RedrawAxis()      #draw axis in foreground
        leg21.Draw("same")
        leg22.Draw("same")
    canvasSum.Print("alltogether.png")



parser = ArgumentParser('import objects from root file')
parser.add_argument("-i", help="set input root file", metavar="FILE")

args=parser.parse_args()

ifile = ROOT.TFile(args.i)

h_ll_pt_data = ifile.Get("ll_pt_data")
h_ll_eta_data = ifile.Get("ll_eta_data")
h_tl_pt_data = ifile.Get("tl_pt_data")
h_tl_eta_data = ifile.Get("tl_eta_data")
h_jet_n = ifile.Get("number_data")

h_data_list = h_ll_pt_data, h_ll_eta_data, h_tl_pt_data, h_tl_eta_data, h_jet_n

hs_ll_pt = ifile.Get("hs;1")
hs_ll_eta = ifile.Get("hs;2")
hs_tl_pt = ifile.Get("hs;3")
hs_tl_eta = ifile.Get("hs;4")
hs_jet_n = ifile.Get("hs;5")

hs_list = hs_ll_pt, hs_ll_eta, hs_tl_pt, hs_tl_eta, hs_jet_n


scaleFactor = h_ll_eta_data.Integral()/ROOT.TH1F(hs_ll_eta.GetStack().Last()).Integral()
print("scaleFactor = ", scaleFactor)
for hs in hs_list:
    for hist in hs.GetHists():
        hist.Scale(scaleFactor)
    hs.Modified()   #actualize the thstack

makeFullPlots(h_data=h_data_list, hs_mc=hs_list, name="all")

pdb.set_trace()

print('end')