from __future__ import print_function

from TrainData import TrainData, fileTimeOut

class TrainData_Discriminator(TrainData):
    '''
    Class used to convert the root files in numpy arrays to train a Discriminator
    Just one label (self.y) which indicates if the event (jet) is simulated (0) or real data (1)
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData.__init__(self)

        self.discriminatortargetclasses='isRealData'
        self.registerBranches([self.discriminatortargetclasses])

        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand', 'nNpfcand',
                          'nsv', 'npv',
                          'TagVarCSV_trackSumJetEtRatio',
                          'TagVarCSV_trackSumJetDeltaR',
                          'TagVarCSV_vertexCategory',
                          'TagVarCSV_trackSip2dValAboveCharm',
                          'TagVarCSV_trackSip2dSigAboveCharm',
                          'TagVarCSV_trackSip3dValAboveCharm',
                          'TagVarCSV_trackSip3dSigAboveCharm',
                          'TagVarCSV_jetNSelectedTracks',
                          'TagVarCSV_jetNTracksEtaRel'
                          ])
        self.addBranches([
                          'Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',

                          'Cpfcan_ptrel',
                          'Cpfcan_drminsv',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw',
                          'Cpfcan_chi2',
                          'Cpfcan_quality'
                          ],
                         25)
        self.addBranches([
                          'Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac',
                          'Npfcan_drminsv',
                          'Npfcan_puppiw'
                          ],
                         25)

        self.addBranches([
                          'sv_pt',
                          'sv_deltaR',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_chi2',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                         4)



    def readFromRootFile(self, filename, TupleMeanStd, weighter):
        from preprocessing import MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from stopwatch import stopwatch

        sw = stopwatch()
        swall = stopwatch()

        import ROOT

        fileTimeOut(filename, 120)  # give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()

        print('took ', sw.getAndReset(), ' seconds for getting tree entries')
        print('number of tree entries: ', self.nsamples)

        # split for convolutional network

        x_global = MeanNormZeroPad(filename, TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]], self.nsamples)

        x_cpf = MeanNormZeroPadParticles(filename, TupleMeanStd,
                                         self.branches[1],
                                         self.branchcutoffs[1], self.nsamples)

        x_npf = MeanNormZeroPadParticles(filename, TupleMeanStd,
                                         self.branches[2],
                                         self.branchcutoffs[2], self.nsamples)

        x_sv = MeanNormZeroPadParticles(filename, TupleMeanStd,
                                        self.branches[3],
                                        self.branchcutoffs[3], self.nsamples)

        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')

        Tuple = self.readTreeFromRootToTuple(filename)




        # I don't know whats happening here, so I switched of the mechanism to remove content
        self.remove=False

        if self.remove:
            notremoves = weighter.createNotRemoveIndices(Tuple)
            undef = Tuple['isUndefined']
            notremoves -= undef
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.weight:
            weights = weighter.getJetWeights(Tuple)
        elif self.remove:
            weights = notremoves
        else:
            print('neither remove nor weight')
            weights = numpy.empty(self.nsamples)
            weights.fill(1.)

        discriminatortargets = Tuple[self.discriminatortargetclasses]


        if self.remove:
            print('remove')
            weights = weights[notremoves > 0]
            x_global = x_global[notremoves > 0]
            x_cpf = x_cpf[notremoves > 0]
            x_npf = x_npf[notremoves > 0]
            x_sv = x_sv[notremoves > 0]


        newnsamp = x_global.shape[0]
        print('reduced content to ', int(float(newnsamp) / float(self.nsamples) * 100), '%')
        self.nsamples = newnsamp

        print('Got content from root file ', filename, ' ...')
        print("     nsamples: ", self.nsamples)
        print("     x_global.shape: ",x_global.shape)
        print("     x_global type: ", x_global.dtype)
        print("     x_cpf.shape: ", x_cpf.shape)
        print("     x_npf.shape: ", x_npf.shape)
        print("     x_sv.shape: ", x_sv.shape)
        print("     discriminatortarget.shape: ", discriminatortargets.shape)

        print('     discriminatortargets type ', discriminatortargets.dtype)


        self.w = [weights]
        self.x = [x_global, x_cpf, x_npf, x_sv]
        self.y = [discriminatortargets]