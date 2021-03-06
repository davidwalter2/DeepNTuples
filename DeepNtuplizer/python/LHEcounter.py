###
#   Program to produce a histogram of the lhe weights to compute the effective event number of a sample with lhe weights
###
import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
### parsing job options
import sys


options = VarParsing.VarParsing()

options.register('inputScript','',VarParsing.VarParsing.multiplicity.singleton,VarParsing.VarParsing.varType.string,"input Script")
options.register('outputFile','output',VarParsing.VarParsing.multiplicity.singleton,VarParsing.VarParsing.varType.string,"output File (w/o .root)")
options.register('maxEvents',-1,VarParsing.VarParsing.multiplicity.singleton,VarParsing.VarParsing.varType.int,"maximum events")
options.register('skipEvents', 0, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "skip N events")
options.register('job', 0, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "job number")
options.register('nJobs', 1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "total jobs")


import os
release=os.environ['CMSSW_VERSION'][6:11]
print("Using release "+release)

options.register(
	'inputFiles','',
	VarParsing.VarParsing.multiplicity.list,
	VarParsing.VarParsing.varType.string,
	"input files "
	)

if hasattr(sys, "argv"):
    options.parseArguments()


process = cms.Process("lheWeightCounter")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 10
if options.inputScript == '': #this is probably for testing
    process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.options = cms.untracked.PSet(
   allowUnscheduled = cms.untracked.bool(True),
   wantSummary=cms.untracked.bool(True)
)


sampleListFile = 'DeepNTuples.DeepNtuplizer.samples.singleMuon_2016_cfg'
process.load(sampleListFile) #default input

if options.inputFiles:
    process.source.fileNames = options.inputFiles

if options.inputScript != '' and options.inputScript != sampleListFile:
    process.load(options.inputScript)

#process.source.fileNames=['file:/afs/cern.ch/work/d/dwalter/data/ttbar/data.root']   #store/data/Run2016H/SingleMuon/MINIAOD/18Apr2017-v1/00000/00E02A09-853C-E711-93FF-3417EBE644A7.root



numberOfFiles = len(process.source.fileNames)
numberOfJobs = options.nJobs
jobNumber = options.job

process.source.fileNames = process.source.fileNames[jobNumber:numberOfFiles:numberOfJobs]
if options.nJobs > 1:
    print ("running over these files:")
    print (process.source.fileNames)

process.source.skipEvents = cms.untracked.uint32(options.skipEvents)
process.maxEvents  = cms.untracked.PSet(
    input = cms.untracked.int32 (options.maxEvents)
)

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)


process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("lheWeights_"+str(options.job) +".root"),
                                   closeFileFast=cms.untracked.bool(True)
                                   )

process.lheAnalyzer = cms.EDAnalyzer('lheAnalyzer',
                               lheInfo = cms.InputTag("externalLHEProducer")
                               )


process.endp = cms.EndPath(process.lheAnalyzer)