'''ROOT drawing helper functions.'''
import ROOT
from ROOT import TLine, TBox, TF1

gc = []

def DrawLine(x0, y0, x1, y1, xrange=None, lw=3, ls='-', color=ROOT.kBlack):
  if x0 > x1: x0, y0, x1, y1 = x1, y1, x0, y0
  if isinstance(xrange, ROOT.TH1):
    xrange = (xrange.GetNbinsX(), xrange.GetBinLowEdge(1),
              xrange.GetBinLowEdge(xrange.GetNbinsX() + 1))

  if isinstance(xrange, (tuple, list)):
    if x0 < xrange[1] and x1 < xrange[1]:
      return None
    elif xrange[2] < x0 and xrange[2] < x1:
      return None
    else:
      if x0 < xrange[1]: x0 = xrange[1]
      if xrange[2] < x1: x1 = xrange[2]

    y0 = y0 * (0.9 - 0.1) + 0.1
    y1 = y1 * (0.9 - 0.1) + 0.1
    x0 = (x0 - xrange[1]) / (xrange[2] - xrange[1]) * (0.9 - 0.1) + 0.1
    x1 = (x1 - xrange[1]) / (xrange[2] - xrange[1]) * (0.9 - 0.1) + 0.1

  l = TLine()
  l.SetLineColor(color)
  l.SetLineWidth(lw)
  if ls == '-':
    l.SetLineStyle(ROOT.kSolid)  # kSolid      = 1
  elif ls == '--':
    l.SetLineStyle(ROOT.kDashed)  # kDashed     = 2
  elif ls == ':':
    l.SetLineStyle(ROOT.kDotted)  # kDotted     = 3
  elif ls == '-.':
    l.SetLineStyle(ROOT.kDashDotted)  # kDashDotted = 4

  l.DrawLineNDC(x0, y0, x1, y1)
  gc.append(l)
  return l


def DrawBox(x0, y0, x1, y1, xrange=None, alpha=0.2, color=ROOT.kBlack):
  if x0 > x1: x0, y0, x1, y1 = x1, y1, x0, y0
  if isinstance(xrange, ROOT.TH1):
    # y0 = xrange.GetYaxis().GetXmin()
    # y1 = xrange.GetYaxis().GetXmax()

    xrange = (xrange.GetNbinsX(), xrange.GetBinLowEdge(1),
              xrange.GetBinLowEdge(xrange.GetNbinsX() + 1))

  if isinstance(xrange, (tuple, list)):
    if x0 < xrange[1] and x1 < xrange[1]:
      return None
    elif xrange[2] < x0 and xrange[2] < x1:
      return None
    else:
      if x0 < xrange[1]: x0 = xrange[1]
      if xrange[2] < x1: x1 = xrange[2]

    # y0 = y0*(0.9-0.1)+0.1
    # y1 = y1*(0.9-0.1)+0.1
    # x0 = (x0 - xrange[1])/(xrange[2] - xrange[1])*(0.9-0.1)+0.1
    # x1 = (x1 - xrange[1])/(xrange[2] - xrange[1])*(0.9-0.1)+0.1

  l = TBox()
  l.SetFillColorAlpha(color, alpha)
  l.DrawBox(x0, y0, x1, y1)

  gc.append(l)
  return l


def Draw(hh, title=None, lw=3, color=ROOT.kAzure, opts='same'):
  if title is not None: hh.SetTitle(title)
  hh.SetLineWidth(lw)
  hh.SetLineColor(color)
  hh.Draw(opts)
  gc.append(hh)


def Draw2D(hh, title=None, opts='colz'):
  if title is not None: hh.SetTitle(title)
  hh.Draw(opts)
  gc.append(hh)


def GausFit(hh,
            gpars=None,
            pol=None,
            srange=3,
            iters=5,
            lw=3,
            colors=(ROOT.kRed, ROOT.kMagenta, ROOT.kGreen + 2),
            draw=True):
  sig = 'gaus(0)'
  bck = 'pol' + str(pol) + '(3)'
  form = sig + ' + ' + bck
  pars = 'gaus(0)'
  if gpars: pars = gpars
  if pol: pars += (pol + 1) * [0]

  for i in range(iters):
    fit = TF1('fit', form, pars[1] - srange * pars[2],
              pars[1] + srange * pars[2])
    fit.SetParameters(*pars)
    # fit.SetParLimits(2, 0, 1e9)
    hh.Fit(fit, 'QNREM')
    pars = [fit.GetParameters()[p] for p in range(len(pars))]
  fit.SetLineWidth(lw)
  fit.SetLineColor(colors[0])

  pars = [fit.GetParameters()[p] for p in range(len(pars))]
  errs = [fit.GetParErrors()[p] for p in range(len(pars))]

  fits = [fit]
  for i, func in enumerate([sig, bck]):
    ff = TF1('ff', func, pars[1] - srange * pars[2],
             pars[1] + srange * pars[2])
    ff.SetLineColor(colors[i + 1])
    ff.SetLineWidth(lw)
    ff.SetParameters(*pars)
    if draw:
      ff.Draw('same')
    fits.append(ff)
    gc.append(ff)

  gc.append(fit)
  return fits, pars, errs
