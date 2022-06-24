"""Main"""
import os
import joblib
import json
import numpy as np
from ROOT import RDataFrame

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.models.autoencoder import VariationalAutoEncoder
from src.models.gan import GenerativeAdversarialNetwork, GANMonitor

from itertools import product


KAON_MASS = 0.493677
PRO_MASS = 0.938272
ELE_MASS = 0.000510999
eps = 1e-8

# feats = [a+b for a in ('e','kp','km') for b in ('x','y','z')]
feats = [a+b for a in ('e','kp','km') for b in ('P','Theta','Phi')]
feats += [a+b for a in ('e','kp','km') for b in ('vx','vy','vz')]
feats += ['mm2', 'phim']
# feats = [a+b for a in ('e','pr','kp','km') for b in ('P','Theta','Phi')]
# feats += [a + b for a in ('e', 'kp', 'km') for b in ('P', 'Theta', 'Phi', 'Pt', 'Eta')]
# feats += ['q2', 't', 'xb', 'nu', 'w']
# feats += ['ekpAngle', 'ekmAngle', 'kpkmAngle', 'mm2', 'phim']
# feats += ['q2', 't', 'xb', 'nu', 'w', 'mm2', 'phim']
# feats += ['ekpAngle', 'ekmAngle', 'eprAngle',
#           'kpkmAngle', 'kpprAngle', 'kmprAngle']

# feats += ['ekpAngle', 'ekmAngle', 'kpkmAngle', 'q2']  #, 'phim']
# feats += ['ekpAngle', 'ekmAngle', 'eprAngle',
#           'kpkmAngle', 'kpprAngle', 'kmprAngle']

def parse_data(fname, test_size=0.2, max_rows=-1):
  # np.random.seed(42)

  print(fname)
  vals = []

  if 'Wagon'in fname:
    rdf = RDataFrame('clas12', fname)
    col_names = list(rdf.GetColumnNames())
    for v in feats:
      if v not in vals and v not in col_names:
        vals.append(v)

    rdf = rdf.Define(
        'vals', '''
    Double_t eps = 1e-8;
    Double_t KAON_MASS = 0.493677, PRO_MASS = 0.938272, ELE_MASS = 0.000510999;
    TLorentzVector beam, pr, targ, ele, kp, km;
    beam.SetXYZM(0, 0, 10.6041, ELE_MASS);
    targ.SetXYZM(0, 0, 0, PRO_MASS);

    ele.SetXYZM(ex, ey, ez, ELE_MASS);
    pr.SetXYZM(px, py, pz, PRO_MASS);
    kp.SetXYZM(kpx, kpy, kpz, KAON_MASS);
    km.SetXYZM(kmx, kmy, kmz, KAON_MASS);

    auto eE = ele.E(), eP = ele.P(), eTheta = ele.Theta(), ePhi = ele.Phi();
    auto kpE = kp.E(), kpP = kp.P(), kpTheta = kp.Theta(), kpPhi = kp.Phi();
    auto kmE = km.E(), kmP = km.P(), kmTheta = km.Theta(), kmPhi = km.Phi();
    auto prE = pr.E(), prP = pr.P(), prTheta = pr.Theta(), prPhi = pr.Phi();

    auto ePt = ele.Pt(), eEta = ele.Eta();
    auto kpPt = kp.Pt(), kpEta = kp.Eta();
    auto kmPt = km.Pt(), kmEta = km.Eta();
    auto prPt = pr.Pt(), prEta = pr.Eta();

    auto ekpAngle = ele.Angle(kp.Vect());
    auto ekmAngle = ele.Angle(km.Vect());
    auto eprAngle = ele.Angle(pr.Vect());
    auto kpkmAngle = kp.Angle(km.Vect());
    auto kpprAngle = kp.Angle(pr.Vect());
    auto kmprAngle = km.Angle(pr.Vect());

    auto phim = (kp+km).M(), mm2 = (beam+targ-ele-kp-km).M2();

    auto q = beam-ele, eX = beam+targ-ele;
    auto q2 = -q.M2();
    auto xb = q2/(2*targ.M()*q.E());
    // auto t  = 2*targ.M()*(pr.E()-targ.M());
    auto t  = (beam - pr).M2();
    auto nu = q2/(2*targ.M()*xb);
    auto y  = nu/beam.E();
    auto w  = eX.M();

    auto lambdam = (beam+targ-ele-kp).M();

    //phim = TMath::Log(phim - 2*KAON_MASS + eps);
    return vector<double>{''' + ','.join(vals) + '};')
    for i, v in enumerate(vals):
      rdf = rdf.Define(v, f'vals[{i}]')

  else:
    rdf = RDataFrame('h22', fname)
    col_names = list(rdf.GetColumnNames())
    for v in feats:
      if v not in vals and v not in col_names:
        print(v)
        vals.append(v)

    rdf = rdf.Define(
        'vals', '''
    Double_t eps = 1e-8;
    Double_t KAON_MASS = 0.493677, PRO_MASS = 0.938272, ELE_MASS = 0.000510999;
    TLorentzVector beam, targ, ele, kp, km;
    beam.SetXYZM(0, 0, 10.6041, ELE_MASS);
    targ.SetXYZM(0, 0, 0, PRO_MASS);
    ele.SetXYZM(ex, ey, ez, ELE_MASS);
    kp.SetXYZM(kpx, kpy, kpz, KAON_MASS);
    km.SetXYZM(kmx, kmy, kmz, KAON_MASS);
    auto pr = beam+targ-ele-kp-km;

    auto eE = ele.E(), eP = ele.P(), eTheta = ele.Theta(), ePhi = ele.Phi();
    auto kpE = kp.E(), kpP = kp.P(), kpTheta = kp.Theta(), kpPhi = kp.Phi();
    auto kmE = km.E(), kmP = km.P(), kmTheta = km.Theta(), kmPhi = km.Phi();
    auto prE = pr.E(), prP = pr.P(), prTheta = pr.Theta(), prPhi = pr.Phi();

    auto ePt = ele.Pt(), eEta = ele.Eta();
    auto kpPt = kp.Pt(), kpEta = kp.Eta();
    auto kmPt = km.Pt(), kmEta = km.Eta();
    auto prPt = pr.Pt(), prEta = pr.Eta();

    auto ekpAngle = ele.Angle(kp.Vect());
    auto ekmAngle = ele.Angle(km.Vect());
    auto eprAngle = ele.Angle(pr.Vect());
    auto kpkmAngle = kp.Angle(km.Vect());
    auto kpprAngle = kp.Angle(pr.Vect());
    auto kmprAngle = km.Angle(pr.Vect());

    auto phim = (kp+km).M(), mm2 = pr.M2();

    auto q = beam-ele, eX = beam+targ-ele;
    auto q2 = -q.M2();
    auto xb = q2/(2*targ.M()*q.E());
    // auto t  = 2*targ.M()*(pr.E()-targ.M());
    auto t  = (beam - pr).M2();
    auto nu = q2/(2*targ.M()*xb);
    auto y  = nu/beam.E();
    auto w  = eX.M();

    auto lambdam = (beam+targ-ele-kp).M();

    //phim = TMath::Log(phim - 2*KAON_MASS + eps);
    return vector<double>{''' + ','.join(vals) + '};')
    for i, v in enumerate(vals):
      rdf = rdf.Define(v, f'vals[{i}]')

  print('Evs -', rdf.Count().GetValue())
  # rdf = rdf.Filter('lambdam < 1.5 || 1.58 < lambdam')
  # rdf = rdf.Filter('lambdam < 1.78 || 1.90 < lambdam')


  if 'kpstat' in col_names:
    print('Evs before FD cut -', rdf.Count().GetValue())
    rdf = rdf.Filter('kmstat < 4000 && kpstat < 4000')
    print('Evs after FD cut -', rdf.Count().GetValue())

  # print('Evs before MM2 6 sigma cut -', rdf.Count().GetValue())
  # mm2_mu, mm2_sg = 0.891369, 0.060926
  # rdf = rdf.Filter(f'{mm2_mu}-6*{mm2_sg} < mm2 && mm2 < {mm2_mu}+6*{mm2_sg}')
  # print('Evs after MM2 6 sigma cut -', rdf.Count().GetValue())

  # phi_mu, phi_sg = 1.020039, 0.004812
  # rdf = rdf.Filter(
  #     f'{phi_mu}-6*{phi_sg} < phim && phim < {phi_mu}+6*{phi_sg}')

  # print('Evs before eta cut -', rdf.Count().GetValue())
  # rdf = rdf.Filter('1.075 < eEta && eEta < 2.783'
  #         ).Filter('0.300 < kpEta && kpEta < 5.116'
  #         ).Filter('0.437 < kmEta && kmEta < 2.979')
  # print('Evs after eta cut -', rdf.Count().GetValue())

  print(feats)
  data_dict = rdf.AsNumpy(feats)
  data = np.stack([data_dict[f] for f in feats]).T

  if max_rows > 0:
    data = data[:max_rows]

  trn, tst = train_test_split(
      data,
      random_state=42,
      test_size=test_size)
      # train_size=100_000)
  print(trn.shape, tst.shape)

  scaler = RobustScaler()
  # scaler = MinMaxScaler(feature_range=(-1,1))
  trn = scaler.fit_transform(trn)
  tst = scaler.transform(tst)

  return trn, tst, scaler


def train_gan(name, trn, tst, scaler, iterations=200_000, **kwargs):
  gan = GenerativeAdversarialNetwork(trn.shape[1], **kwargs)
  gan.compile()

  batch_size = 32
  epochs = int(iterations // (len(trn) / batch_size) + 1)

  out_dir = f'models/{name}'
  os.system(f'mkdir -p {out_dir}')
  joblib.dump(scaler, f'{out_dir}/scaler.joblib')

  callbacks = [
    GANMonitor(training_data=train,validation_data=tst),
    TensorBoard(log_dir=f'logs/{name}'),
    ModelCheckpoint(filepath = out_dir+'/ckpt/epoch{epoch}'),
    # EarlyStopping(monitor='val_rmse',
    #               mode='min',
    #               patience=15,
    #               restore_best_weights=True),
  ]

  hh = gan.fit(trn,
               batch_size=batch_size,
               epochs=epochs,
               callbacks=callbacks,
               verbose=1)#0: silent, 1: progress bar, 2: one line/epoch

  gan.save(f'{out_dir}/GAN', save_format='tf')

  with open(f'{out_dir}/history', 'w', encoding='utf-8') as ff:
    json.dump(str(hh.history), ff)

  return gan

def train_vae(name, trn, tst, scaler, iterations=200_000, batch_size=128, **kwargs):
  vae = VariationalAutoEncoder(trn.shape[1], **kwargs)
  vae.compile(optimizer='adam')
  # vae.compile(optimizer=Adam(learning_rate=0.0001, clipvalue=0.5))

  if iterations < 1000:
    epochs = iterations
  else:
    epochs = int(iterations // (len(trn) / batch_size) + 1)

  out_dir = f'models/{name}'
  os.system(f'mkdir -p {out_dir}')
  joblib.dump(scaler, f'{out_dir}/scaler.joblib')

  callbacks = [
    TensorBoard(log_dir=f'logs/{name}'),
    ModelCheckpoint(filepath = out_dir+'/ckpt/epoch{epoch}'),
    # EarlyStopping(monitor='val_rmse',
    #               mode='min',
    #               patience=15,
    #               restore_best_weights=True),
  ]

  hh = vae.fit(trn,
               batch_size=batch_size,
               epochs=epochs,
               callbacks=callbacks,
               verbose=1)#0: silent, 1: progress bar, 2: one line/epoch

  vae.save(f'{out_dir}/VAE', save_format='tf')

  with open(f'{out_dir}/history', 'w', encoding='utf-8') as ff:
    json.dump(str(hh.history), ff)

  return vae

if __name__ == '__main__':
  os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
  train, test, sclr = parse_data('data/external/DVPhiWagon_ntuple_inb.root', test_size=2)
  # train, test, sclr = parse_data('data/external/eKpKm_fa2018_sp2019.root')

  # train, test, sclr = parse_data('data/external/gen_rdf.root')
  # train, test, sclr = parse_data('data/external/simu_eKpKm_fa2018.root')
  for i in range(train.shape[1]):
    if np.isnan(train[:, i]).any():
      print(f'Error with {i}')
      raise Exception(f'Error {i}!')


  for vari in (True, ):#False):
    for ld in (2, 4, 6):
      name = f'{"v" if vari else ""}ae_LD{ld}'

      print(f'\nStaring {name}')
      mdl = train_vae(name,
                      train,
                      test,
                      sclr,
                      iterations=200,
                      batch_size=32,
                      latent_dim=ld,
                      variational=vari)


  # nlays = (2, 4, 8)
  # norms = (None, 'batch', 'layer')
  # drops = (None, 1, 2, 3, 4)
  # iterables = np.array(list(product(norms, norms, drops, drops)))
  # np.random.shuffle(iterables)

  # for iters in [('batch', 'layer', None, 5), ('batch', 'layer', None, 2)]:#iterables:
  #   gNorm, dNorm, gDO, dDO = iters

  #   if gDO or gNorm is None or dNorm == 'batch':
  #     continue

  #   name = 'model_small'
  #   if gDO or gNorm:
  #     name += '_g'
  #     if gDO:
  #       name += f'DO{gDO}'
  #       gDO /= 10
  #     if gNorm:
  #       if gNorm == 'batch':
  #         name += 'BN'
  #       elif gNorm == 'layer':
  #         name += 'LN'
  #   if dDO or dNorm:
  #     name += '_d'
  #     if dDO:
  #       name += f'DO{dDO}'
  #       dDO /= 10
  #     if dNorm:
  #       if dNorm == 'batch':
  #         name += 'BN'
  #       elif dNorm == 'layer':
  #         name += 'LN'

  #   print(f'\nStaring {name}')
  #   mdl = train_gan(name,
  #                   train,
  #                   test,
  #                   sclr,
  #                   iterations=200_000,
  #                   gen_layers=2,
  #                   dis_layers=4,
  #                   gen_normalization=gNorm,
  #                   gen_drop_rate=gDO,
  #                   dis_normalization=dNorm,
  #                   dis_drop_rate=dDO)
