from testing import makeROCs_async

infile='tree_association.txt'

outdir = ''
makeROCs_async(intextfile=infile,
               name_list=['DeepFlavour'],
               probabilities_list='prob_isB+prob_isBB+prob_isLeptB',
               truths_list='isB+isBB+isGBB + isLeptonicB+ isLeptonicB_C',
               vetos_list=1*['isUD+isS+isG']+1*['isC+isCC+isGCC'],
               colors_list='auto',
               outpdffile=outdir+"bJets.pdf",
               #cuts='jet_pt>30',
               cmsstyle=True,
               firstcomment='W+Jets events',
               #secondcomment='jet p_{T} > 30 GeV',
               extralegend=['solid?udsg','dashed?c'],
               logY=False)