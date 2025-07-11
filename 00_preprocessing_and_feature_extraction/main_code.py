# -*- coding: utf-8 -*-
"""
Created on Nov 25 07:49:59 2023

@author: Personal
"""

"""

runfile('C:/Users/idrobo/Documents/Ennio/1 SIESTA-vs-VitalDB-Project/Scripts/main_code.py', wdir='C:/Users/idrobo/Documents/Ennio/1 SIESTA-vs-VitalDB-Project/Scripts', args='--start 1 --stop 2')
"""


#%% Importing libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import vitaldb
import code_functions
from code_functions import signal_shannon_entropy, quality_signal, quality_multi_signal
from code_functions import notch_filter, golay_filter,low_pass_filter, bandpass, saavedra,R_correction
from code_functions import correccion_picuda,detect_bestAlgorithm
from code_functions import correction_outliers, interpolacion_vbeats,cambio_porcentaje_nn, interpolacion_final
from code_functions import band_pass, inter_tacograma,detect_picos_olvidados,interpolacion_bis
from code_functions import timedom_hrv,frequency_hrv, time_domain, caoticas,caoticas2,caract_entropia, bandpower, rms
from code_functions import freq_domain , mutual_information, calculate_nonlinear_interdependence
from scipy import interpolate
from scipy.interpolate import interp1d
import argparse
from pathlib import Path  
import glob, os
from scipy import signal
from vitaldb import VitalFile
def main_processing(startindex, stopindex):
    
   
    #%% Preparation
    # df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # clinical information
    # df_trks  = pd.read_csv("https://api.vitaldb.net/trks")  # track list
    # df_labs  = pd.read_csv('https://api.vitaldb.net/labs')  # laboratory results
    
    df_cases = pd.read_csv('/hpc/gpfs2/scratch/u/idroboen/viltadb/physionet.org/files/vitaldb/1.0.0/clinical_data.csv')
    df_trks  = pd.read_csv("/hpc/gpfs2/scratch/u/idroboen/viltadb/physionet.org/files/vitaldb/1.0.0/df_trks.csv")  # track list
    # df_labs  = pd.read_csv('lab_data.csv')  # laboratory results
    
    # Replacing values ">89" to'90'
    df_cases.age = df_cases.age.replace(">89",'90')
    df_cases.age = pd.to_numeric(df_cases.age)

    #%% Inclusion & Exclusion Criteria
    '''
    SNUADC/ART, BIS/EEG1_WAV, BIS/EEG2_WAV, and Primus/CO2 refer to the device used for recording the data.
    The last line is for including subjects in a supine position.
    '''
    caseids = list(
    set(df_trks.loc[df_trks['tname'] == 'BIS/EEG1_WAV', 'caseid']) &
    set(df_trks.loc[df_trks['tname'] == 'BIS/EEG2_WAV', 'caseid']) &
    #set(df_trks.loc[df_trks['tname'] == 'SNUADC/ECG_V5', 'caseid']) &
    set(df_trks.loc[df_trks['tname'] == 'SNUADC/ECG_II', 'caseid']) &
    set(df_trks.loc[df_trks['tname'] == 'BIS/BIS','caseid']) &
    set(df_cases.loc[df_cases['department'] == 'General surgery','caseid']) &  #Tipo de cirugia
    set(df_cases.loc[df_cases['ane_type'] == 'General','caseid'])&  # Tipo de anestesia
    set(df_cases.loc[df_cases['position'] == 'Supine','caseid']) & # Posición boca arriba
    set(df_cases.loc[df_cases['approach'] == 'Open','caseid']) & # Abordaje quirurgico: abierto
    set(df_cases.loc[((df_cases['age'])>19) & (df_cases['age']<71) ,'caseid'])& # Pacientes con rango de edad 20-70 años
    set(df_cases.loc[df_cases['bmi']<=24.9,'caseid'])& # pacientes con peso normal
    set(df_cases.loc[df_cases['preop_pft']=='Normal','caseid'])& #pacientes con funcion pulmonar normal
    (set(df_cases.loc[df_cases['asa'] == 1, 'caseid']) | set(df_cases.loc[df_cases['asa'] == 2, 'caseid'])) & #pacientes asa 
    set(df_cases.loc[df_cases['preop_htn']==0,'caseid'])&
    set(df_cases.loc[df_cases['preop_dm']==0,'caseid'])&
    set(df_cases.loc[df_cases['preop_ecg']=='Normal Sinus Rhythm','caseid']) # pacientes ritmo normal
    )    
   
    
    print(f'{len(caseids)} cases found')
    # 1620




    # %% Reading all cases


    for caseid_indx in range(startindex, stopindex, 1):
        print('caseid_indx: ---------------------------------------------------------------')
        print(caseid_indx)
    
        Fs_eeg = 250
        Fs_ecg = 250
       
        filename = str(caseids[caseid_indx]).zfill(4) + '.vital'
        vf = vitaldb.vital_recs("/hpc/gpfs2/scratch/u/idroboen/viltadb/physionet.org/files/vitaldb/1.0.0/vital_files/"+filename)

        vals = vf.to_pandas(['BIS/EEG1_WAV', 'BIS/EEG2_WAV'], interval = 1/Fs_eeg)
        vals2 = vf.to_pandas(['SNUADC/ECG_II'], interval = 1/Fs_ecg)

        df_bis = vf.to_pandas(['BIS/BIS'], return_datetime=True, interval=1)
    
        try:
            # Assigning signals
            signal_eeg1 = vals['BIS/EEG1_WAV'].values
            signal_eeg2 = vals['BIS/EEG2_WAV'].values
            signal_ecgII = vals2['SNUADC/ECG_II'].values
            df_bis['Seconds'] = df_bis['Time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
            BIS = df_bis['BIS/BIS'].tolist()
            time_bis  = df_bis['Seconds'].tolist()
            cond = np.any(BIS)
            if len(BIS) == 0 or all(val == BIS[0] for val in BIS) or cond == False or np.size(BIS) == 0:
                continue
            
        except:
            continue
    
    
        Fs = Fs_eeg
        samp_rates  = [Fs, Fs, Fs]
        signals     = [signal_eeg1, signal_eeg2, signal_ecgII]
        det_segms   = [False, False, False]
        segm_time   = 5
        window_time = 60
        overlap = False
        labels_q_L, signal_0_ind, signal_1_ind, signal_2_ind = quality_multi_signal(samp_rates, signals, det_segms, segm_time, window_time,overlap)
    
        # Para todas las señales es el mismo ind_start_good y ind_end_good
        # Escojo solo el de la señal eeg1
        ind_start_good = signal_0_ind["ind_start_good_L0"]
        ind_end_good = signal_0_ind["ind_end_good_L0"]
    
        #Datos demograficos:
        filtered_data = df_cases.loc[df_cases['caseid'] == caseids[caseid_indx]][['caseid', 'casestart', 'caseend', 'anestart', 'aneend', 'opstart', 'opend', 'age', 'sex', 'height', 'weight', 'optype', 'dx']]
    
        data_list = []
        for indice in range(0,len(ind_end_good),1):
            #Tiempo del segmento - donde inicia y termina // para todas las señales es la misma
            tiempo_start=(ind_start_good[indice])*(1/Fs)
            tiempo_end=(ind_end_good[indice])*(1/Fs)
            t=np.arange(tiempo_start,tiempo_end,1/Fs)
    
            #Segmentos EEG1
            signal_segm_eeg1=signal_eeg1[ind_start_good[indice]:ind_end_good[indice]]
            signal_segm_eeg1 = signal_segm_eeg1.astype(np.float64)
            signal_notch_eeg1 = notch_filter(signal_segm_eeg1)
            signal_golay_eeg1 = golay_filter(signal_notch_eeg1,'db4', 2)
    
            #Segmentos EEG2
            signal_segm_eeg2=signal_eeg2[ind_start_good[indice]:ind_end_good[indice]]
            signal_segm_eeg2 = signal_segm_eeg2.astype(np.float64)
            signal_notch_eeg2 = notch_filter(signal_segm_eeg2)
            signal_golay_eeg2 = golay_filter(signal_notch_eeg2,'db4', 2)
    
            #Segmentos ECGII
            signal_segm_ecgII=signal_ecgII[ind_start_good[indice]:ind_end_good[indice]]
            signal_segm_ecgII = signal_segm_ecgII.astype(np.float64)
            signal_notch_ecgII = notch_filter(signal_segm_ecgII)
            signal_golay_ecgII = golay_filter(signal_notch_ecgII,'bior3.5', 2)
            seg_ecg_blw_2 = low_pass_filter(signal_golay_ecgII, 0.45)
    
            # filtro para detectar el complejo QRS
            seg_qrs_2 = bandpass(seg_ecg_blw_2)
    
            #Encontrar HRV
            try:
                #detect_bestAlgorithm
                best_algorithm_2,r_pks_2,r_pks_ms_2 = detect_bestAlgorithm(seg_qrs_2)
            except:
                continue
    
            
            try:

                if len(r_pks_2)<40:
                    continue
    
                nn_2=correction_outliers(r_pks_2,len(r_pks_2),t, 5.2)
    
            except:
                continue
    
            #Encontrar segmentos de cada paciente de la señal BIS
            #Interpolar valores nan y ceros
            bis_inter = interpolacion_bis(BIS)
            #interpolar para obtener señal del mismo tamaño
            fcubic = interpolate.interp1d(time_bis, bis_inter, kind='cubic')
            tnew = np.arange(time_bis[0], time_bis[-1], 1/250)
            if tnew[-1] > np.max(time_bis): 
                tnew[-1] = np.max(time_bis)
            ycubic = fcubic(tnew)
            tiempo_start_bis=int(tiempo_start)
            tiempo_end_bis=int(tiempo_end)
    
            muestra_start=(tiempo_start_bis*Fs)
            muestra_end=tiempo_end_bis*Fs
            seg_bis=ycubic[muestra_start:muestra_end]
    
            #Definir estados de anestesia
            estados = [1] * len(ind_start_good)
            mean_bis_seg = np.mean(seg_bis)
    
    
            if 80 <= mean_bis_seg <= 100:
              estados[indice] = 0
            elif 60 <= mean_bis_seg <= 80:
              estados[indice] = 1
            elif 40 <= mean_bis_seg <= 60:
              estados[indice] = 2
            elif 20 <= mean_bis_seg <= 40:
              estados[indice] = 3
            elif 0 <= mean_bis_seg <= 20:
              estados[indice] = 4
    
    
    
            # Encontrar outliers e interpolar nn
            nn_2 =nn_2*1000
            val_beats2 =cambio_porcentaje_nn(nn_2)
            nn_inter2 = interpolacion_final(val_beats2,nn_2)
             
            if len(nn_inter2) == 0 or np.all(nn_inter2 == nn_inter2[0]): 
                continue
    
            #Descomponer señal EEG en diferentes bandas de frecuencia
            eeg1_lenta = band_pass(signal_golay_eeg1,0.1,1,250)
            eeg1_delta = band_pass(signal_golay_eeg1,1,4,250)
            eeg1_theta = band_pass(signal_golay_eeg1,5,8,250)
            eeg1_alpha = band_pass(signal_golay_eeg1,9,12,250)
            eeg1_beta = band_pass(signal_golay_eeg1,13,25,250)
            eeg1_gamma = band_pass(signal_golay_eeg1,26,80,250)
            
            eeg2_lenta = band_pass(signal_golay_eeg2,0.1,1,250)
            eeg2_delta = band_pass(signal_golay_eeg2,1,4,250)
            eeg2_theta = band_pass(signal_golay_eeg2,5,8,250)
            eeg2_alpha = band_pass(signal_golay_eeg2,9,12,250)
            eeg2_beta = band_pass(signal_golay_eeg2,13,25,250)
            eeg2_gamma = band_pass(signal_golay_eeg2,26,80,250)
            
            #Descomponer señal ECG en diferentes bandas de frecuencia
            ecg2_lenta = band_pass(seg_ecg_blw_2,0.1,1,250)
            ecg2_delta = band_pass(seg_ecg_blw_2,1,4,250)
            ecg2_theta = band_pass(seg_ecg_blw_2,5,8,250)
            ecg2_alpha = band_pass(seg_ecg_blw_2,9,12,250)
            ecg2_beta = band_pass(seg_ecg_blw_2,13,25,250)
            ecg2_gamma = band_pass(seg_ecg_blw_2,26,80,250)
            
            #Descomponer señal HRV en diferentes bandas de frecuencia
            #  vlf_band=(0.003, 0.04), lf_band=(0.04, 0.15), hf_band=(0.15, 0.4))
            
            tacograma_hf = band_pass(inter_tacograma(nn_inter2,180),0.15,0.4,4)
            tacograma_lf = band_pass(inter_tacograma(nn_inter2,180),0.04,0.15,4)
            #tacograma_vlf = band_pass(inter_tacograma(nn_inter2),0.003,0.04,50)
            
            # Caracteristicas EEG
    
            
            # Tiempo:
            #EEG
            time_eeg1 = time_domain(signal_golay_eeg1)
            time_eeg1_lenta = time_domain(eeg1_lenta)
            time_eeg1_delta = time_domain(eeg1_delta)
            time_eeg1_theta = time_domain(eeg1_theta)
            time_eeg1_alpha = time_domain(eeg1_alpha)
            time_eeg1_beta = time_domain(eeg1_beta)
            time_eeg1_gamma = time_domain(eeg1_gamma)
            
            time_eeg2 = time_domain(signal_golay_eeg2)
            time_eeg2_lenta = time_domain(eeg2_lenta)
            time_eeg2_delta = time_domain(eeg2_delta)
            time_eeg2_theta = time_domain(eeg2_theta)
            time_eeg2_alpha = time_domain(eeg2_alpha)
            time_eeg2_beta = time_domain(eeg2_beta)
            time_eeg2_gamma = time_domain(eeg2_gamma)
            
            # Frecuencia: trans_fourtier
            freq_eeg1 = freq_domain(signal_golay_eeg1)
            freq_eeg1_lenta = freq_domain(eeg1_lenta)
            freq_eeg1_delta = freq_domain(eeg1_delta)
            freq_eeg1_theta = freq_domain(eeg1_theta)
            freq_eeg1_alpha = freq_domain(eeg1_alpha)
            freq_eeg1_beta = freq_domain(eeg1_beta)
            freq_eeg1_gamma = freq_domain(eeg1_gamma)
            
            freq_eeg2 = freq_domain(signal_golay_eeg2)
            freq_eeg2_lenta = freq_domain(eeg2_lenta)
            freq_eeg2_delta = freq_domain(eeg2_delta)
            freq_eeg2_theta = freq_domain(eeg2_theta)
            freq_eeg2_alpha = freq_domain(eeg2_alpha)
            freq_eeg2_beta = freq_domain(eeg2_beta)
            freq_eeg2_gamma = freq_domain(eeg2_gamma)
            
            #BIS
            time_bis_ = time_domain(seg_bis)
            
            # Entropia:
            ent_eeg1 = caract_entropia(signal_golay_eeg1)
            ent_eeg1_lenta = caract_entropia(eeg1_lenta)
            ent_eeg1_delta = caract_entropia(eeg1_delta)
            ent_eeg1_theta = caract_entropia(eeg1_theta)
            ent_eeg1_alpha = caract_entropia(eeg1_alpha)
            ent_eeg1_beta = caract_entropia(eeg1_beta)
            ent_eeg1_gamma = caract_entropia(eeg1_gamma)
            
            ent_eeg2 = caract_entropia(signal_golay_eeg2)
            ent_eeg2_lenta = caract_entropia(eeg2_lenta)
            ent_eeg2_delta = caract_entropia(eeg2_delta)
            ent_eeg2_theta = caract_entropia(eeg2_theta)
            ent_eeg2_alpha = caract_entropia(eeg2_alpha)
            ent_eeg2_beta = caract_entropia(eeg2_beta)
            ent_eeg2_gamma = caract_entropia(eeg2_gamma)
            
            # Caoticas:
            cao_eeg1 = caoticas(signal_golay_eeg1)
            cao_eeg1_lenta = caoticas(eeg1_lenta)
            cao_eeg1_delta = caoticas(eeg1_delta)
            cao_eeg1_theta = caoticas(eeg1_theta)
            cao_eeg1_alpha = caoticas(eeg1_alpha)
            cao_eeg1_beta = caoticas(eeg1_beta)
            cao_eeg1_gamma = caoticas(eeg1_gamma)
            
            cao_eeg2 = caoticas(signal_golay_eeg2)
            cao_eeg2_lenta = caoticas(eeg2_lenta)
            cao_eeg2_delta = caoticas(eeg2_delta)
            cao_eeg2_theta = caoticas(eeg2_theta)
            cao_eeg2_alpha = caoticas(eeg2_alpha)
            cao_eeg2_beta = caoticas(eeg2_beta)
            cao_eeg2_gamma = caoticas(eeg2_gamma)
            
            # Caracteristicas ECG
            
           
            # Tiempo:
            time_ecg2 = time_domain(seg_ecg_blw_2)
            time_ecg2_lenta = time_domain(ecg2_lenta)
            time_ecg2_delta = time_domain(ecg2_delta)
            time_ecg2_theta = time_domain(ecg2_theta)
            time_ecg2_alpha = time_domain(ecg2_alpha)
            time_ecg2_beta = time_domain(ecg2_beta)
            time_ecg2_gamma = time_domain(ecg2_gamma)
            
            #Frecuencia:
            freq_ecg2 = freq_domain(seg_ecg_blw_2)
            freq_ecg2_lenta = freq_domain(ecg2_lenta)
            freq_ecg2_delta = freq_domain(ecg2_delta)
            freq_ecg2_theta = freq_domain(ecg2_theta)
            freq_ecg2_alpha = freq_domain(ecg2_alpha)
            freq_ecg2_beta = freq_domain(ecg2_beta)
            freq_ecg2_gamma = freq_domain(ecg2_gamma)
            
            #Entropia:
            ent_ecg2 = caract_entropia(seg_ecg_blw_2)
            ent_ecg2_lenta = caract_entropia(ecg2_lenta)
            ent_ecg2_delta = caract_entropia(ecg2_delta)
            ent_ecg2_theta = caract_entropia(ecg2_theta)
            ent_ecg2_alpha = caract_entropia(ecg2_alpha)
            ent_ecg2_beta = caract_entropia(ecg2_beta)
            ent_ecg2_gamma = caract_entropia(ecg2_gamma)
            
            # Caoticas:
            cao_ecg2 = caoticas(seg_ecg_blw_2)
            cao_ecg2_lenta = caoticas(ecg2_lenta)
            cao_ecg2_delta = caoticas(ecg2_delta)
            cao_ecg2_theta = caoticas(ecg2_theta)
            cao_ecg2_alpha = caoticas(ecg2_alpha)
            cao_ecg2_beta = caoticas(ecg2_beta)
            cao_ecg2_gamma = caoticas(ecg2_gamma)
            
            
            # Caracteristicas HRV
            time_hrv = timedom_hrv(inter_tacograma(nn_inter2,180))
            frec_hrv = frequency_hrv(inter_tacograma(nn_inter2,180))
            
            # Caracteristicas hrv, hf y lf
            # Tiempo:
            time_hf =  time_domain(tacograma_hf)
            time_lf =  time_domain(tacograma_lf)
            
            # Entropia:
            ent_hf  = caract_entropia(tacograma_hf)
            ent_lf  = caract_entropia(tacograma_lf)
            ent_hrv  = caract_entropia(nn_inter2)
            
            # Caoticas:
            cao_hf  = caoticas2(tacograma_hf)
            cao_lf  = caoticas2(tacograma_lf)
            cao_hrv  = caoticas2(inter_tacograma(nn_inter2,180))
                            
                
            #Band power
            win = 4* 250
            bp_eeg1_lenta = bandpower(signal_golay_eeg1, 250,[0.1,1],win)
            bp_eeg1_delta = bandpower(signal_golay_eeg1, 250,[1,4],win)
            bp_eeg1_theta = bandpower(signal_golay_eeg1, 250,[5,8],win)
            bp_eeg1_alpha = bandpower(signal_golay_eeg1, 250,[9,12],win)
            bp_eeg1_beta = bandpower(signal_golay_eeg1, 250,[13,25],win)
            bp_eeg1_gamma = bandpower(signal_golay_eeg1, 250,[26,80],win)
            
            bp_eeg2_lenta = bandpower(signal_golay_eeg2, 250,[0.1,1],win)
            bp_eeg2_delta = bandpower(signal_golay_eeg2, 250,[1,4],win)
            bp_eeg2_theta = bandpower(signal_golay_eeg2, 250,[5,8],win)
            bp_eeg2_alpha = bandpower(signal_golay_eeg2, 250,[9,12],win)
            bp_eeg2_beta = bandpower(signal_golay_eeg2, 250,[13,25],win)
            bp_eeg2_gamma = bandpower(signal_golay_eeg2, 250,[26,80],win)
            
            bp_ecg2_lenta = bandpower(seg_ecg_blw_2, 250,[0.1,1],win)
            bp_ecg2_delta = bandpower(seg_ecg_blw_2, 250,[1,4],win)
            bp_ecg2_theta = bandpower(seg_ecg_blw_2, 250,[5,8],win)
            bp_ecg2_alpha = bandpower(seg_ecg_blw_2, 250,[9,12],win)
            bp_ecg2_beta = bandpower(seg_ecg_blw_2, 250,[13,25],win)
            bp_ecg2_gamma = bandpower(seg_ecg_blw_2, 250,[26,80],win)
            
            #RMS
            #Señales
            rms_eeg1_lenta = rms(eeg1_lenta)
            rms_eeg1_delta = rms(eeg1_delta)
            rms_eeg1_theta = rms(eeg1_theta)
            rms_eeg1_alpha = rms(eeg1_alpha)
            rms_eeg1_beta = rms(eeg1_beta)
            rms_eeg1_gamma = rms(eeg1_gamma)
            
            rms_eeg2_lenta = rms(eeg2_lenta)
            rms_eeg2_delta = rms(eeg2_delta)
            rms_eeg2_theta = rms(eeg2_theta)
            rms_eeg2_alpha = rms(eeg2_alpha)
            rms_eeg2_beta = rms(eeg2_beta)
            rms_eeg2_gamma = rms(eeg2_gamma)
            
            rms_ecg2_lenta = rms(ecg2_lenta)
            rms_ecg2_delta = rms(ecg2_delta)
            rms_ecg2_theta = rms(ecg2_theta)
            rms_ecg2_alpha = rms(ecg2_alpha)
            rms_ecg2_beta = rms(ecg2_beta)
            rms_ecg2_gamma = rms(ecg2_gamma)
            
            #Espectros
            sf = 250
            win = 4 * sf
            freqs1, psd1= signal.welch(signal_golay_eeg1, sf, nperseg=win)
            freqs2, psd2 = signal.welch(signal_golay_eeg2, sf, nperseg=win)
            freqs3, psd3 = signal.welch(seg_ecg_blw_2, sf, nperseg=win)
            
            esp_rms_eeg1 = rms (psd1)
            esp_rms_eeg1_lenta = rms(bp_eeg1_lenta[2])
            esp_rms_eeg1_delta = rms(bp_eeg1_delta[2])
            esp_rms_eeg1_theta = rms(bp_eeg1_theta[2])
            esp_rms_eeg1_alpha = rms(bp_eeg1_alpha[2])
            esp_rms_eeg1_beta = rms(bp_eeg1_beta[2])
            esp_rms_eeg1_gamma = rms(bp_eeg1_gamma[2])
            
            esp_rms_eeg2 = rms (psd2)
            esp_rms_eeg2_lenta = rms(bp_eeg2_lenta[2])
            esp_rms_eeg2_delta = rms(bp_eeg2_delta[2])
            esp_rms_eeg2_theta = rms(bp_eeg2_theta[2])
            esp_rms_eeg2_alpha = rms(bp_eeg2_alpha[2])
            esp_rms_eeg2_beta = rms(bp_eeg2_beta[2])
            esp_rms_eeg2_gamma = rms(bp_eeg2_gamma[2])
            
            esp_rms_ecg2 = rms (psd3)
            esp_rms_ecg2_lenta = rms(bp_ecg2_lenta[2])
            esp_rms_ecg2_delta = rms(bp_ecg2_delta[2])
            esp_rms_ecg2_theta = rms(bp_ecg2_theta[2])
            esp_rms_ecg2_alpha = rms(bp_ecg2_alpha[2])
            esp_rms_ecg2_beta = rms(bp_ecg2_beta[2])
            esp_rms_ecg2_gamma = rms(bp_ecg2_gamma[2])
            
            #Heart rate
            NBEATS = len(nn_inter2)
            hr = [1] * NBEATS
            
            for i in range(0, NBEATS, 1):
                hr[i] = 60000 / nn_inter2[i]
    
            time_hr = time_domain(hr)
            current_data_dict = filtered_data.iloc[0].to_dict()
            
            
            # Agregar más claves al diccionario si es necesario
            current_data_dict.update({
                'Estado': estados[indice],
                
                #BIS
                'bis_max':time_bis_[0], 'bis_min':time_bis_[1], 'bis_med':time_bis_[2],
                'bis_median':time_bis_[3], 'bis_mod':time_bis_[4], 'bis_var':time_bis_[5],
                'bis_std':time_bis_[6], 
                
                #Tiempo 
                'tiempo_start': tiempo_start,
                'tiempo_end':tiempo_end,
                
                #Heart rate
                'hr_max':time_hr[0], 'hr_min':time_hr[1], 'hr_med':time_hr[2],
                'hr_median':time_hr[3], 'hr_mod':time_hr[4], 'hr_var':time_hr[5],
                'hr_std':time_hr[6], 
            
                #EEG dominio del tiempo y frecuencia
                #EEG1
                'eeg1_max':time_eeg1[0], 'eeg1_min':time_eeg1[1], 'eeg1_med':time_eeg1[2],
                'eeg1_median':time_eeg1[3], 'eeg1_mod':time_eeg1[4], 'eeg1_var':time_eeg1[5],
                'eeg1_std':time_eeg1[6], 'eeg1_pow':time_eeg1[7] , 'eeg1_tasa':time_eeg1[8],
                'eeg1_curt':time_eeg1[9],'eeg1_ske':time_eeg1[10],
                
                'eeg1_max_freq': freq_eeg1[0], 'eeg1_min_freq': freq_eeg1[1], 'eeg1_med_freq': freq_eeg1[2],
                'eeg1_median_freq': freq_eeg1[3], 'eeg1_mod_freq': freq_eeg1[4], 'eeg1_var_freq': freq_eeg1[5],
                'eeg1_std_freq': freq_eeg1[6], 'eeg1_pow_freq': freq_eeg1[7], 'eeg1_tasa_freq': freq_eeg1[8],
                'eeg1_curt_freq': freq_eeg1[9], 'eeg1_ske_freq': freq_eeg1[10], 'eeg1_freq_max': freq_eeg1[11], 
                'eeg1_mean_freq': freq_eeg1[12], 'eeg1_freq_median': freq_eeg1[13], 'eeg1_freq_centroid': freq_eeg1[14], 
                'eeg1_freq_dispersion': freq_eeg1[15], 'eeg1_freq_flatness': freq_eeg1[16], 'eeg1_freq_slope': freq_eeg1[17], 
                'eeg1_freq_crest_factor': freq_eeg1[18], 
                
                'eeg1_lenta_max':time_eeg1_lenta[0], 'eeg1_lenta_min':time_eeg1_lenta[1], 'eeg1_lenta_med':time_eeg1_lenta[2],
                'eeg1_lenta_median':time_eeg1_lenta[3], 'eeg1_lenta_mod':time_eeg1_lenta[4], 'eeg1_lenta_var':time_eeg1_lenta[5],
                'eeg1_lenta_std':time_eeg1_lenta[6], 'eeg1_lenta_pow':time_eeg1_lenta[7] , 'eeg1_lenta_tasa':time_eeg1_lenta[8],
                'eeg1_lenta_curt':time_eeg1_lenta[9],'eeg1_lenta_ske':time_eeg1_lenta[10],
                
                'eeg1_lenta_max_freq': freq_eeg1_lenta[0], 'eeg1_lenta_min_freq': freq_eeg1_lenta[1], 'eeg1_lenta_med_freq': freq_eeg1_lenta[2],
                'eeg1_lenta_median_freq': freq_eeg1_lenta[3], 'eeg1_lenta_mod_freq': freq_eeg1_lenta[4], 'eeg1_lenta_var_freq': freq_eeg1_lenta[5],
                'eeg1_lenta_std_freq': freq_eeg1_lenta[6], 'eeg1_lenta_pow_freq': freq_eeg1_lenta[7], 'eeg1_lenta_tasa_freq': freq_eeg1_lenta[8],
                'eeg1_lenta_curt_freq': freq_eeg1_lenta[9], 'eeg1_lenta_ske_freq': freq_eeg1_lenta[10],'eeg1_lenta_freq_max': freq_eeg1_lenta[11], 
                'eeg1_lenta_mean_freq': freq_eeg1_lenta[12], 'eeg1_lenta_freq_median': freq_eeg1_lenta[13], 'eeg1_lenta_freq_centroid': freq_eeg1_lenta[14], 
                'eeg1_lenta_freq_dispersion': freq_eeg1_lenta[15], 'eeg1_lenta_freq_flatness': freq_eeg1_lenta[16], 'eeg1_lenta_freq_slope': freq_eeg1_lenta[17], 
                'eeg1_lenta_freq_crest_factor': freq_eeg1_lenta[18],

            
                'eeg1_delta_max':time_eeg1_delta[0], 'eeg1_delta_min':time_eeg1_delta[1], 'eeg1_delta_med':time_eeg1_delta[2],
                'eeg1_delta_median':time_eeg1_delta[3], 'eeg1_delta_mod':time_eeg1_delta[4], 'eeg1_delta_var':time_eeg1_delta[5],
                'eeg1_delta_std':time_eeg1_delta[6], 'eeg1_delta_pow':time_eeg1_delta[7] , 'eeg1_delta_tasa':time_eeg1_delta[8],
                'eeg1_delta_curt':time_eeg1_delta[9],'eeg1_delta_ske':time_eeg1_delta[10],
                
                'eeg1_delta_max_freq': freq_eeg1_delta[0], 'eeg1_delta_min_freq': freq_eeg1_delta[1], 'eeg1_delta_med_freq': freq_eeg1_delta[2],
                'eeg1_delta_median_freq': freq_eeg1_delta[3], 'eeg1_delta_mod_freq': freq_eeg1_delta[4], 'eeg1_delta_var_freq': freq_eeg1_delta[5],
                'eeg1_delta_std_freq': freq_eeg1_delta[6], 'eeg1_delta_pow_freq': freq_eeg1_delta[7], 'eeg1_delta_tasa_freq': freq_eeg1_delta[8],
                'eeg1_delta_curt_freq': freq_eeg1_delta[9], 'eeg1_delta_ske_freq': freq_eeg1_delta[10],
                'eeg1_delta_freq_max': freq_eeg1_delta[11], 'eeg1_delta_mean_freq': freq_eeg1_delta[12], 
                'eeg1_delta_freq_median': freq_eeg1_delta[13], 'eeg1_delta_freq_centroid': freq_eeg1_delta[14],
                'eeg1_delta_freq_dispersion': freq_eeg1_delta[15], 'eeg1_delta_freq_flatness': freq_eeg1_delta[16],
                'eeg1_delta_freq_slope': freq_eeg1_delta[17], 'eeg1_delta_freq_crest_factor': freq_eeg1_delta[18],
            
                'eeg1_theta_max':time_eeg1_theta[0], 'eeg1_theta_min':time_eeg1_theta[1], 'eeg1_theta_med':time_eeg1_theta[2],
                'eeg1_theta_median':time_eeg1_theta[3], 'eeg1_theta_mod':time_eeg1_theta[4], 'eeg1_theta_var':time_eeg1_theta[5],
                'eeg1_theta_std':time_eeg1_theta[6], 'eeg1_theta_pow':time_eeg1_theta[7] , 'eeg1_theta_tasa':time_eeg1_theta[8],
                'eeg1_theta_curt':time_eeg1_theta[9],'eeg1_theta_ske':time_eeg1_theta[10],
            
                'eeg1_theta_max_freq': freq_eeg1_theta[0], 'eeg1_theta_min_freq': freq_eeg1_theta[1], 'eeg1_theta_med_freq': freq_eeg1_theta[2],
                'eeg1_theta_median_freq': freq_eeg1_theta[3], 'eeg1_theta_mod_freq': freq_eeg1_theta[4], 'eeg1_theta_var_freq': freq_eeg1_theta[5],
                'eeg1_theta_std_freq': freq_eeg1_theta[6], 'eeg1_theta_pow_freq': freq_eeg1_theta[7], 'eeg1_theta_tasa_freq': freq_eeg1_theta[8],
                'eeg1_theta_curt_freq': freq_eeg1_theta[9], 'eeg1_theta_ske_freq': freq_eeg1_theta[10],'eeg1_theta_freq_max': freq_eeg1_theta[11],
                'eeg1_theta_mean_freq': freq_eeg1_theta[12], 'eeg1_theta_freq_median': freq_eeg1_theta[13], 'eeg1_theta_freq_centroid': freq_eeg1_theta[14],
                'eeg1_theta_freq_dispersion': freq_eeg1_theta[15], 'eeg1_theta_freq_flatness': freq_eeg1_theta[16], 'eeg1_theta_freq_slope': freq_eeg1_theta[17], 
                'eeg1_theta_freq_crest_factor': freq_eeg1_theta[18],
            
                'eeg1_alpha_max':time_eeg1_alpha[0], 'eeg1_alpha_min':time_eeg1_alpha[1], 'eeg1_alpha_med':time_eeg1_alpha[2],
                'eeg1_alpha_median':time_eeg1_alpha[3], 'eeg1_alpha_mod':time_eeg1_alpha[4], 'eeg1_alpha_var':time_eeg1_alpha[5],
                'eeg1_alpha_std':time_eeg1_alpha[6], 'eeg1_alpha_pow':time_eeg1_alpha[7] , 'eeg1_alpha_tasa':time_eeg1_alpha[8],
                'eeg1_alpha_curt':time_eeg1_alpha[9],'eeg1_alpha_ske':time_eeg1_alpha[10],
                
                'eeg1_alpha_max_freq': freq_eeg1_alpha[0], 'eeg1_alpha_min_freq': freq_eeg1_alpha[1], 'eeg1_alpha_med_freq': freq_eeg1_alpha[2],
                'eeg1_alpha_median_freq': freq_eeg1_alpha[3], 'eeg1_alpha_mod_freq': freq_eeg1_alpha[4], 'eeg1_alpha_var_freq': freq_eeg1_alpha[5],
                'eeg1_alpha_std_freq': freq_eeg1_alpha[6], 'eeg1_alpha_pow_freq': freq_eeg1_alpha[7], 'eeg1_alpha_tasa_freq': freq_eeg1_alpha[8],
                'eeg1_alpha_curt_freq': freq_eeg1_alpha[9], 'eeg1_alpha_ske_freq': freq_eeg1_alpha[10],'eeg1_alpha_freq_max': freq_eeg1_alpha[11], 
                'eeg1_alpha_mean_freq': freq_eeg1_alpha[12], 'eeg1_alpha_freq_median': freq_eeg1_alpha[13], 'eeg1_alpha_freq_centroid': freq_eeg1_alpha[14],
                'eeg1_alpha_freq_dispersion': freq_eeg1_alpha[15], 'eeg1_alpha_freq_flatness': freq_eeg1_alpha[16], 'eeg1_alpha_freq_slope': freq_eeg1_alpha[17], 
                'eeg1_alpha_freq_crest_factor': freq_eeg1_alpha[18],

            
                'eeg1_beta_max':time_eeg1_beta[0], 'eeg1_beta_min':time_eeg1_beta[1], 'eeg1_beta_med':time_eeg1_beta[2],
                'eeg1_beta_median':time_eeg1_beta[3], 'eeg1_beta_mod':time_eeg1_beta[4], 'eeg1_beta_var':time_eeg1_beta[5],
                'eeg1_beta_std':time_eeg1_beta[6], 'eeg1_beta_pow':time_eeg1_beta[7] , 'eeg1_beta_tasa':time_eeg1_beta[8],
                'eeg1_beta_curt':time_eeg1_beta[9],'eeg1_beta_ske':time_eeg1_beta[10],
                
                'eeg1_beta_max_freq': freq_eeg1_beta[0], 'eeg1_beta_min_freq': freq_eeg1_beta[1], 'eeg1_beta_med_freq': freq_eeg1_beta[2],
                'eeg1_beta_median_freq': freq_eeg1_beta[3], 'eeg1_beta_mod_freq': freq_eeg1_beta[4], 'eeg1_beta_var_freq': freq_eeg1_beta[5],
                'eeg1_beta_std_freq': freq_eeg1_beta[6], 'eeg1_beta_pow_freq': freq_eeg1_beta[7], 'eeg1_beta_tasa_freq': freq_eeg1_beta[8],
                'eeg1_beta_curt_freq': freq_eeg1_beta[9], 'eeg1_beta_ske_freq': freq_eeg1_beta[10], 'eeg1_beta_freq_max': freq_eeg1_beta[11], 
                'eeg1_beta_mean_freq': freq_eeg1_beta[12], 'eeg1_beta_freq_median': freq_eeg1_beta[13], 'eeg1_beta_freq_centroid': freq_eeg1_beta[14],
                'eeg1_beta_freq_dispersion': freq_eeg1_beta[15], 'eeg1_beta_freq_flatness': freq_eeg1_beta[16], 'eeg1_beta_freq_slope': freq_eeg1_beta[17],
                'eeg1_beta_freq_crest_factor': freq_eeg1_beta[18],

            
                'eeg1_gamma_max': time_eeg1_gamma[0], 'eeg1_gamma_min': time_eeg1_gamma[1], 'eeg1_gamma_med': time_eeg1_gamma[2],
                'eeg1_gamma_median': time_eeg1_gamma[3], 'eeg1_gamma_mod': time_eeg1_gamma[4], 'eeg1_gamma_var': time_eeg1_gamma[5],
                'eeg1_gamma_std': time_eeg1_gamma[6], 'eeg1_gamma_pow': time_eeg1_gamma[7], 'eeg1_gamma_tasa': time_eeg1_gamma[8],
                'eeg1_gamma_curt': time_eeg1_gamma[9], 'eeg1_gamma_ske': time_eeg1_gamma[10],
            
                'eeg1_gamma_max_freq': freq_eeg1_gamma[0], 'eeg1_gamma_min_freq': freq_eeg1_gamma[1], 'eeg1_gamma_med_freq': freq_eeg1_gamma[2],
                'eeg1_gamma_median_freq': freq_eeg1_gamma[3], 'eeg1_gamma_mod_freq': freq_eeg1_gamma[4], 'eeg1_gamma_var_freq': freq_eeg1_gamma[5],
                'eeg1_gamma_std_freq': freq_eeg1_gamma[6], 'eeg1_gamma_pow_freq': freq_eeg1_gamma[7], 'eeg1_gamma_tasa_freq': freq_eeg1_gamma[8],
                'eeg1_gamma_curt_freq': freq_eeg1_gamma[9], 'eeg1_gamma_ske_freq': freq_eeg1_gamma[10], 'eeg1_gamma_freq_max': freq_eeg1_gamma[11], 
                'eeg1_gamma_mean_freq': freq_eeg1_gamma[12], 'eeg1_gamma_freq_median': freq_eeg1_gamma[13], 'eeg1_gamma_freq_centroid': freq_eeg1_gamma[14], 
                'eeg1_gamma_freq_dispersion': freq_eeg1_gamma[15], 'eeg1_gamma_freq_flatness': freq_eeg1_gamma[16], 'eeg1_gamma_freq_slope': freq_eeg1_gamma[17],
                'eeg1_gamma_freq_crest_factor': freq_eeg1_gamma[18],


            
                #EEG2 dominio del tiempo y frecuencia
                'eeg2_max':time_eeg2[0], 'eeg2_min':time_eeg2[1], 'eeg2_med':time_eeg2[2],
                'eeg2_median':time_eeg2[3], 'eeg2_mod':time_eeg2[4], 'eeg2_var':time_eeg2[5],
                'eeg2_std':time_eeg2[6], 'eeg2_pow':time_eeg2[7] , 'eeg2_tasa':time_eeg2[8],
                'eeg2_curt':time_eeg2[9],'eeg2_ske':time_eeg2[10],
                
                              
                'eeg2_max_freq': freq_eeg2[0], 'eeg2_min_freq': freq_eeg2[1], 'eeg2_med_freq': freq_eeg2[2], 'eeg2_median_freq': freq_eeg2[3], 
                'eeg2_mod_freq': freq_eeg2[4], 'eeg2_var_freq': freq_eeg2[5], 'eeg2_std_freq': freq_eeg2[6], 'eeg2_pow_freq': freq_eeg2[7], 
                'eeg2_tasa_freq': freq_eeg2[8], 'eeg2_curt_freq': freq_eeg2[9], 'eeg2_ske_freq': freq_eeg2[10], 'eeg2_freq_max': freq_eeg2[11], 
                'eeg2_mean_freq': freq_eeg2[12], 'eeg2_freq_median': freq_eeg2[13], 'eeg2_freq_centroid': freq_eeg2[14], 'eeg2_freq_dispersion': freq_eeg2[15], 
                'eeg2_freq_flatness': freq_eeg2[16], 'eeg2_freq_slope': freq_eeg2[17], 'eeg2_freq_crest_factor': freq_eeg2[18],

               
                'eeg2_lenta_max': time_eeg2_lenta[0], 'eeg2_lenta_min': time_eeg2_lenta[1], 'eeg2_lenta_med': time_eeg2_lenta[2],
                'eeg2_lenta_median': time_eeg2_lenta[3], 'eeg2_lenta_mod': time_eeg2_lenta[4], 'eeg2_lenta_var': time_eeg2_lenta[5],
                'eeg2_lenta_std': time_eeg2_lenta[6], 'eeg2_lenta_pow': time_eeg2_lenta[7], 'eeg2_lenta_tasa': time_eeg2_lenta[8],
                'eeg2_lenta_curt':time_eeg2_lenta[9],'eeg2_lenta_ske':time_eeg2_lenta[10],
            
                'eeg2_lenta_max_freq': freq_eeg2_lenta[0], 'eeg2_lenta_min_freq': freq_eeg2_lenta[1], 'eeg2_lenta_med_freq': freq_eeg2_lenta[2], 
                'eeg2_lenta_median_freq': freq_eeg2_lenta[3], 'eeg2_lenta_mod_freq': freq_eeg2_lenta[4], 'eeg2_lenta_var_freq': freq_eeg2_lenta[5], 
                'eeg2_lenta_std_freq': freq_eeg2_lenta[6], 'eeg2_lenta_pow_freq': freq_eeg2_lenta[7], 'eeg2_lenta_tasa_freq': freq_eeg2_lenta[8], 
                'eeg2_lenta_curt_freq': freq_eeg2_lenta[9], 'eeg2_lenta_ske_freq': freq_eeg2_lenta[10], 'eeg2_lenta_freq_max': freq_eeg2_lenta[11], 
                'eeg2_lenta_mean_freq': freq_eeg2_lenta[12], 'eeg2_lenta_freq_median': freq_eeg2_lenta[13], 'eeg2_lenta_freq_centroid': freq_eeg2_lenta[14], 
                'eeg2_lenta_freq_dispersion': freq_eeg2_lenta[15], 'eeg2_lenta_freq_flatness': freq_eeg2_lenta[16], 'eeg2_lenta_freq_slope': freq_eeg2_lenta[17], 
                'eeg2_lenta_freq_crest_factor': freq_eeg2_lenta[18],

            
                'eeg2_delta_max': time_eeg2_delta[0], 'eeg2_delta_min': time_eeg2_delta[1], 'eeg2_delta_med': time_eeg2_delta[2],
                'eeg2_delta_median': time_eeg2_delta[3], 'eeg2_delta_mod': time_eeg2_delta[4], 'eeg2_delta_var': time_eeg2_delta[5],
                'eeg2_delta_std': time_eeg2_delta[6], 'eeg2_delta_pow': time_eeg2_delta[7], 'eeg2_delta_tasa': time_eeg2_delta[8],
                'eeg2_delta_curt':time_eeg2_delta[9],'eeg2_delta_ske':time_eeg2_delta[10],
            
                'eeg2_delta_max_freq': freq_eeg2_delta[0], 'eeg2_delta_min_freq': freq_eeg2_delta[1], 'eeg2_delta_med_freq': freq_eeg2_delta[2], 
                'eeg2_delta_median_freq': freq_eeg2_delta[3], 'eeg2_delta_mod_freq': freq_eeg2_delta[4], 'eeg2_delta_var_freq': freq_eeg2_delta[5], 
                'eeg2_delta_std_freq': freq_eeg2_delta[6], 'eeg2_delta_pow_freq': freq_eeg2_delta[7], 'eeg2_delta_tasa_freq': freq_eeg2_delta[8], 
                'eeg2_delta_curt_freq': freq_eeg2_delta[9], 'eeg2_delta_ske_freq': freq_eeg2_delta[10], 'eeg2_delta_freq_max': freq_eeg2_delta[11], 
                'eeg2_delta_mean_freq': freq_eeg2_delta[12], 'eeg2_delta_freq_median': freq_eeg2_delta[13], 'eeg2_delta_freq_centroid': freq_eeg2_delta[14], 
                'eeg2_delta_freq_dispersion': freq_eeg2_delta[15], 'eeg2_delta_freq_flatness': freq_eeg2_delta[16], 'eeg2_delta_freq_slope': freq_eeg2_delta[17], 
                'eeg2_delta_freq_crest_factor': freq_eeg2_delta[18],


                'eeg2_theta_max': time_eeg2_theta[0], 'eeg2_theta_min': time_eeg2_theta[1], 'eeg2_theta_med': time_eeg2_theta[2],
                'eeg2_theta_median': time_eeg2_theta[3], 'eeg2_theta_mod': time_eeg2_theta[4], 'eeg2_theta_var': time_eeg2_theta[5],
                'eeg2_theta_std': time_eeg2_theta[6], 'eeg2_theta_pow': time_eeg2_theta[7], 'eeg2_theta_tasa': time_eeg2_theta[8],
                'eeg2_theta_curt':time_eeg2_theta[9],'eeg2_theta_ske':time_eeg2_theta[10],
            
                'eeg2_theta_max_freq': freq_eeg2_theta[0], 'eeg2_theta_min_freq': freq_eeg2_theta[1], 'eeg2_theta_med_freq': freq_eeg2_theta[2], 
                'eeg2_theta_median_freq': freq_eeg2_theta[3], 'eeg2_theta_mod_freq': freq_eeg2_theta[4], 'eeg2_theta_var_freq': freq_eeg2_theta[5], 
                'eeg2_theta_std_freq': freq_eeg2_theta[6], 'eeg2_theta_pow_freq': freq_eeg2_theta[7], 'eeg2_theta_tasa_freq': freq_eeg2_theta[8], 
                'eeg2_theta_curt_freq': freq_eeg2_theta[9], 'eeg2_theta_ske_freq': freq_eeg2_theta[10], 'eeg2_theta_freq_max': freq_eeg2_theta[11], 
                'eeg2_theta_mean_freq': freq_eeg2_theta[12], 'eeg2_theta_freq_median': freq_eeg2_theta[13], 'eeg2_theta_freq_centroid': freq_eeg2_theta[14], 
                'eeg2_theta_freq_dispersion': freq_eeg2_theta[15], 'eeg2_theta_freq_flatness': freq_eeg2_theta[16], 'eeg2_theta_freq_slope': freq_eeg2_theta[17], 
                'eeg2_theta_freq_crest_factor': freq_eeg2_theta[18],

            
                'eeg2_alpha_max': time_eeg2_alpha[0], 'eeg2_alpha_min': time_eeg2_alpha[1], 'eeg2_alpha_med': time_eeg2_alpha[2],
                'eeg2_alpha_median': time_eeg2_alpha[3], 'eeg2_alpha_mod': time_eeg2_alpha[4], 'eeg2_alpha_var': time_eeg2_alpha[5],
                'eeg2_alpha_std': time_eeg2_alpha[6], 'eeg2_alpha_pow': time_eeg2_alpha[7], 'eeg2_alpha_tasa': time_eeg2_alpha[8],
                'eeg2_alpha_curt':time_eeg2_alpha[9],'eeg2_alpha_ske':time_eeg2_alpha[10],
                
                'eeg2_alpha_max_freq': freq_eeg2_alpha[0], 'eeg2_alpha_min_freq': freq_eeg2_alpha[1], 'eeg2_alpha_med_freq': freq_eeg2_alpha[2], 
                'eeg2_alpha_median_freq': freq_eeg2_alpha[3], 'eeg2_alpha_mod_freq': freq_eeg2_alpha[4], 'eeg2_alpha_var_freq': freq_eeg2_alpha[5], 
                'eeg2_alpha_std_freq': freq_eeg2_alpha[6], 'eeg2_alpha_pow_freq': freq_eeg2_alpha[7], 'eeg2_alpha_tasa_freq': freq_eeg2_alpha[8], 
                'eeg2_alpha_curt_freq': freq_eeg2_alpha[9], 'eeg2_alpha_ske_freq': freq_eeg2_alpha[10], 'eeg2_alpha_freq_max': freq_eeg2_alpha[11], 
                'eeg2_alpha_mean_freq': freq_eeg2_alpha[12], 'eeg2_alpha_freq_median': freq_eeg2_alpha[13], 'eeg2_alpha_freq_centroid': freq_eeg2_alpha[14], 
                'eeg2_alpha_freq_dispersion': freq_eeg2_alpha[15], 'eeg2_alpha_freq_flatness': freq_eeg2_alpha[16], 'eeg2_alpha_freq_slope': freq_eeg2_alpha[17], 
                'eeg2_alpha_freq_crest_factor': freq_eeg2_alpha[18],

           
                'eeg2_beta_max': time_eeg2_beta[0], 'eeg2_beta_min': time_eeg2_beta[1], 'eeg2_beta_med': time_eeg2_beta[2],
                'eeg2_beta_median': time_eeg2_beta[3], 'eeg2_beta_mod': time_eeg2_beta[4], 'eeg2_beta_var': time_eeg2_beta[5],
                'eeg2_beta_std': time_eeg2_beta[6], 'eeg2_abeta_pow': time_eeg2_beta[7], 'eeg2_beta_tasa': time_eeg2_beta[8],
                'eeg2_beta_curt':time_eeg2_beta[9],'eeg2_beta_ske':time_eeg2_beta[10],
                
                'eeg2_beta_max_freq': freq_eeg2_beta[0], 'eeg2_beta_min_freq': freq_eeg2_beta[1], 'eeg2_beta_med_freq': freq_eeg2_beta[2], 
                'eeg2_beta_median_freq': freq_eeg2_beta[3], 'eeg2_beta_mod_freq': freq_eeg2_beta[4], 'eeg2_beta_var_freq': freq_eeg2_beta[5], 
                'eeg2_beta_std_freq': freq_eeg2_beta[6], 'eeg2_beta_pow_freq': freq_eeg2_beta[7], 'eeg2_beta_tasa_freq': freq_eeg2_beta[8], 
                'eeg2_beta_curt_freq': freq_eeg2_beta[9], 'eeg2_beta_ske_freq': freq_eeg2_beta[10], 'eeg2_beta_freq_max': freq_eeg2_beta[11], 
                'eeg2_beta_mean_freq': freq_eeg2_beta[12], 'eeg2_beta_freq_median': freq_eeg2_beta[13], 'eeg2_beta_freq_centroid': freq_eeg2_beta[14], 
                'eeg2_beta_freq_dispersion': freq_eeg2_beta[15], 'eeg2_beta_freq_flatness': freq_eeg2_beta[16], 'eeg2_beta_freq_slope': freq_eeg2_beta[17], 
                'eeg2_beta_freq_crest_factor': freq_eeg2_beta[18],

                'eeg2_gamma_max': time_eeg2_gamma[0], 'eeg2_gamma_min': time_eeg2_gamma[1], 'eeg2_gamma_med': time_eeg2_gamma[2],
                'eeg2_gamma_median': time_eeg2_gamma[3], 'eeg2_gamma_mod': time_eeg2_gamma[4], 'eeg2_gamma_var': time_eeg2_gamma[5],
                'eeg2_gamma_std': time_eeg2_gamma[6], 'eeg2_gamma_pow': time_eeg2_gamma[7], 'eeg2_gamma_tasa': time_eeg2_gamma[8],
                'eeg2_gamma_curt': time_eeg2_gamma[9], 'eeg2_gamma_ske': time_eeg2_gamma[10],
            
                'eeg2_gamma_max_freq': freq_eeg2_gamma[0], 'eeg2_gamma_min_freq': freq_eeg2_gamma[1], 'eeg2_gamma_med_freq': freq_eeg2_gamma[2], 
                'eeg2_gamma_median_freq': freq_eeg2_gamma[3], 'eeg2_gamma_mod_freq': freq_eeg2_gamma[4], 'eeg2_gamma_var_freq': freq_eeg2_gamma[5], 
                'eeg2_gamma_std_freq': freq_eeg2_gamma[6], 'eeg2_gamma_pow_freq': freq_eeg2_gamma[7], 'eeg2_gamma_tasa_freq': freq_eeg2_gamma[8], 
                'eeg2_gamma_curt_freq': freq_eeg2_gamma[9], 'eeg2_gamma_ske_freq': freq_eeg2_gamma[10], 'eeg2_gamma_freq_max': freq_eeg2_gamma[11], 
                'eeg2_gamma_mean_freq': freq_eeg2_gamma[12], 'eeg2_gamma_freq_median': freq_eeg2_gamma[13], 'eeg2_gamma_freq_centroid': freq_eeg2_gamma[14], 
                'eeg2_gamma_freq_dispersion': freq_eeg2_gamma[15], 'eeg2_gamma_freq_flatness': freq_eeg2_gamma[16], 'eeg2_gamma_freq_slope': freq_eeg2_gamma[17], 
                'eeg2_gamma_freq_crest_factor': freq_eeg2_gamma[18],


                #ECG2 dominio del tiempo y frecuencia
                'ecg2_max':time_ecg2[0], 'ecg2_min':time_ecg2[1], 'ecg2_med':time_ecg2[2],
                'ecg2_median':time_ecg2[3], 'ecg2_mod':time_ecg2[4], 'ecg2_var':time_ecg2[5],
                'ecg2_std':time_ecg2[6], 'ecg2_pow':time_ecg2[7] , 'ecg2_tasa':time_ecg2[8],
                'ecg2_curt':time_ecg2[9],'ecg2_ske':time_ecg2[10],
                
                'ecg2_max_freq': freq_ecg2[0], 'ecg2_min_freq': freq_ecg2[1], 'ecg2_med_freq': freq_ecg2[2], 'ecg2_median_freq': freq_ecg2[3], 
                'ecg2_mod_freq': freq_ecg2[4], 'ecg2_var_freq': freq_ecg2[5], 'ecg2_std_freq': freq_ecg2[6], 'ecg2_pow_freq': freq_ecg2[7], 
                'ecg2_tasa_freq': freq_ecg2[8], 'ecg2_curt_freq': freq_ecg2[9], 'ecg2_ske_freq': freq_ecg2[10], 'ecg2_freq_max': freq_ecg2[11], 
                'ecg2_mean_freq': freq_ecg2[12], 'ecg2_freq_median': freq_ecg2[13], 'ecg2_freq_centroid': freq_ecg2[14], 'ecg2_freq_dispersion': freq_ecg2[15], 
                'ecg2_freq_flatness': freq_ecg2[16], 'ecg2_freq_slope': freq_ecg2[17], 'ecg2_freq_crest_factor': freq_ecg2[18],

               
                'ecg2_lenta_max': time_ecg2_lenta[0], 'ecg2_lenta_min': time_ecg2_lenta[1], 'ecg2_lenta_med': time_ecg2_lenta[2],
                'ecg2_lenta_median': time_ecg2_lenta[3], 'ecg2_lenta_mod': time_ecg2_lenta[4], 'ecg2_lenta_var': time_ecg2_lenta[5],
                'ecg2_lenta_std': time_ecg2_lenta[6], 'ecg2_lenta_pow': time_ecg2_lenta[7], 'ecg2_lenta_tasa': time_ecg2_lenta[8],
                'ecg2_lenta_curt':time_ecg2_lenta[9],'ecg2_lenta_ske':time_ecg2_lenta[10],
            
                'ecg2_lenta_max_freq': freq_ecg2_lenta[0], 'ecg2_lenta_min_freq': freq_ecg2_lenta[1], 'ecg2_lenta_med_freq': freq_ecg2_lenta[2], 
                'ecg2_lenta_median_freq': freq_ecg2_lenta[3], 'ecg2_lenta_mod_freq': freq_ecg2_lenta[4], 'ecg2_lenta_var_freq': freq_ecg2_lenta[5], 
                'ecg2_lenta_std_freq': freq_ecg2_lenta[6], 'ecg2_lenta_pow_freq': freq_ecg2_lenta[7], 'ecg2_lenta_tasa_freq': freq_ecg2_lenta[8], 
                'ecg2_lenta_curt_freq': freq_ecg2_lenta[9], 'ecg2_lenta_ske_freq': freq_ecg2_lenta[10], 'ecg2_lenta_freq_max': freq_ecg2_lenta[11], 
                'ecg2_lenta_mean_freq': freq_ecg2_lenta[12], 'ecg2_lenta_freq_median': freq_ecg2_lenta[13], 'ecg2_lenta_freq_centroid': freq_ecg2_lenta[14], 
                'ecg2_lenta_freq_dispersion': freq_ecg2_lenta[15], 'ecg2_lenta_freq_flatness': freq_ecg2_lenta[16], 'ecg2_lenta_freq_slope': freq_ecg2_lenta[17], 
                'ecg2_lenta_freq_crest_factor': freq_ecg2_lenta[18],
           
                'ecg2_delta_max': time_ecg2_delta[0], 'ecg2_delta_min': time_ecg2_delta[1], 'ecg2_delta_med': time_ecg2_delta[2],
                'ecg2_delta_median': time_ecg2_delta[3], 'ecg2_delta_mod': time_ecg2_delta[4], 'ecg2_delta_var': time_ecg2_delta[5],
                'ecg2_delta_std': time_ecg2_delta[6], 'ecg2_delta_pow': time_ecg2_delta[7], 'ecg2_delta_tasa': time_ecg2_delta[8],
                'ecg2_delta_curt':time_ecg2_delta[9],'ecg2_delta_ske':time_ecg2_delta[10],
            
                'ecg2_delta_max_freq': freq_ecg2_delta[0], 'ecg2_delta_min_freq': freq_ecg2_delta[1], 'ecg2_delta_med_freq': freq_ecg2_delta[2], 
                'ecg2_delta_median_freq': freq_ecg2_delta[3], 'ecg2_delta_mod_freq': freq_ecg2_delta[4], 'ecg2_delta_var_freq': freq_ecg2_delta[5], 
                'ecg2_delta_std_freq': freq_ecg2_delta[6], 'ecg2_delta_pow_freq': freq_ecg2_delta[7], 'ecg2_delta_tasa_freq': freq_ecg2_delta[8], 
                'ecg2_delta_curt_freq': freq_ecg2_delta[9], 'ecg2_delta_ske_freq': freq_ecg2_delta[10], 'ecg2_delta_freq_max': freq_ecg2_delta[11], 
                'ecg2_delta_mean_freq': freq_ecg2_delta[12], 'ecg2_delta_freq_median': freq_ecg2_delta[13], 'ecg2_delta_freq_centroid': freq_ecg2_delta[14], 
                'ecg2_delta_freq_dispersion': freq_ecg2_delta[15], 'ecg2_delta_freq_flatness': freq_ecg2_delta[16], 'ecg2_delta_freq_slope': freq_ecg2_delta[17], 
                'ecg2_delta_freq_crest_factor': freq_ecg2_delta[18],
                
                'ecg2_theta_max': time_ecg2_theta[0], 'ecg2_theta_min': time_ecg2_theta[1], 'ecg2_theta_med': time_ecg2_theta[2],
                'ecg2_theta_median': time_ecg2_theta[3], 'ecg2_theta_mod': time_ecg2_theta[4], 'ecg2_theta_var': time_ecg2_theta[5],
                'ecg2_theta_std': time_ecg2_theta[6], 'ecg2_theta_pow': time_ecg2_theta[7], 'ecg2_theta_tasa': time_ecg2_theta[8],
                'ecg2_theta_curt':time_ecg2_theta[9],'ecg2_theta_ske':time_ecg2_theta[10],
                
                'ecg2_theta_max_freq': freq_ecg2_theta[0], 'ecg2_theta_min_freq': freq_ecg2_theta[1], 'ecg2_theta_med_freq': freq_ecg2_theta[2], 
                'ecg2_theta_median_freq': freq_ecg2_theta[3], 'ecg2_theta_mod_freq': freq_ecg2_theta[4], 'ecg2_theta_var_freq': freq_ecg2_theta[5], 
                'ecg2_theta_std_freq': freq_ecg2_theta[6], 'ecg2_theta_pow_freq': freq_ecg2_theta[7], 'ecg2_theta_tasa_freq': freq_ecg2_theta[8], 
                'ecg2_theta_curt_freq': freq_ecg2_theta[9], 'ecg2_theta_ske_freq': freq_ecg2_theta[10], 'ecg2_theta_freq_max': freq_ecg2_theta[11], 
                'ecg2_theta_mean_freq': freq_ecg2_theta[12], 'ecg2_theta_freq_median': freq_ecg2_theta[13], 'ecg2_theta_freq_centroid': freq_ecg2_theta[14], 
                'ecg2_theta_freq_dispersion': freq_ecg2_theta[15], 'ecg2_theta_freq_flatness': freq_ecg2_theta[16], 'ecg2_theta_freq_slope': freq_ecg2_theta[17], 
                'ecg2_theta_freq_crest_factor': freq_ecg2_theta[18],
            
                'ecg2_alpha_max': time_ecg2_alpha[0], 'ecg2_alpha_min': time_ecg2_alpha[1], 'ecg2_alpha_med': time_ecg2_alpha[2],
                'ecg2_alpha_median': time_ecg2_alpha[3], 'ecg2_alpha_mod': time_ecg2_alpha[4], 'ecg2_alpha_var': time_ecg2_alpha[5],
                'ecg2_alpha_std': time_ecg2_alpha[6], 'ecg2_alpha_pow': time_ecg2_alpha[7], 'ecg2_alpha_tasa': time_ecg2_alpha[8],
                'ecg2_alpha_curt':time_eeg2_alpha[9],'ecg2_alpha_ske':time_ecg2_alpha[10],
                
                'ecg2_alpha_max_freq': freq_ecg2_alpha[0], 'ecg2_alpha_min_freq': freq_ecg2_alpha[1], 'ecg2_alpha_med_freq': freq_ecg2_alpha[2], 
                'ecg2_alpha_median_freq': freq_ecg2_alpha[3], 'ecg2_alpha_mod_freq': freq_ecg2_alpha[4], 'ecg2_alpha_var_freq': freq_ecg2_alpha[5], 
                'ecg2_alpha_std_freq': freq_ecg2_alpha[6], 'ecg2_alpha_pow_freq': freq_ecg2_alpha[7], 'ecg2_alpha_tasa_freq': freq_ecg2_alpha[8], 
                'ecg2_alpha_curt_freq': freq_ecg2_alpha[9], 'ecg2_alpha_ske_freq': freq_ecg2_alpha[10], 'ecg2_alpha_freq_max': freq_ecg2_alpha[11], 
                'ecg2_alpha_mean_freq': freq_ecg2_alpha[12], 'ecg2_alpha_freq_median': freq_ecg2_alpha[13], 'ecg2_alpha_freq_centroid': freq_ecg2_alpha[14], 
                'ecg2_alpha_freq_dispersion': freq_ecg2_alpha[15], 'ecg2_alpha_freq_flatness': freq_ecg2_alpha[16], 'ecg2_alpha_freq_slope': freq_ecg2_alpha[17], 
                'ecg2_alpha_freq_crest_factor': freq_ecg2_alpha[18],
           
                'ecg2_beta_max': time_ecg2_beta[0], 'ecg2_beta_min': time_ecg2_beta[1], 'ecg2_beta_med': time_ecg2_beta[2],
                'ecg2_beta_median': time_ecg2_beta[3], 'ecg2_beta_mod': time_ecg2_beta[4], 'ecg2_beta_var': time_ecg2_beta[5],
                'ecg2_beta_std': time_ecg2_beta[6], 'ecg2_abeta_pow': time_ecg2_beta[7], 'ecg2_beta_tasa': time_ecg2_beta[8],
                'ecg2_beta_curt':time_ecg2_beta[9],'ecg2_beta_ske':time_ecg2_beta[10],
                
                'ecg2_beta_max_freq': freq_ecg2_beta[0], 'ecg2_beta_min_freq': freq_ecg2_beta[1], 'ecg2_beta_med_freq': freq_ecg2_beta[2], 
                'ecg2_beta_median_freq': freq_ecg2_beta[3], 'ecg2_beta_mod_freq': freq_ecg2_beta[4], 'ecg2_beta_var_freq': freq_ecg2_beta[5], 
                'ecg2_beta_std_freq': freq_ecg2_beta[6], 'ecg2_beta_pow_freq': freq_ecg2_beta[7], 'ecg2_beta_tasa_freq': freq_ecg2_beta[8], 
                'ecg2_beta_curt_freq': freq_ecg2_beta[9], 'ecg2_beta_ske_freq': freq_ecg2_beta[10], 'ecg2_beta_freq_max': freq_ecg2_beta[11], 
                'ecg2_beta_mean_freq': freq_ecg2_beta[12], 'ecg2_beta_freq_median': freq_ecg2_beta[13], 'ecg2_beta_freq_centroid': freq_ecg2_beta[14], 
                'ecg2_beta_freq_dispersion': freq_ecg2_beta[15], 'ecg2_beta_freq_flatness': freq_ecg2_beta[16], 'ecg2_beta_freq_slope': freq_ecg2_beta[17], 
                'ecg2_beta_freq_crest_factor': freq_ecg2_beta[18],

                'ecg2_gamma_max': time_ecg2_gamma[0], 'ecg2_gamma_min': time_ecg2_gamma[1], 'ecg2_gamma_med': time_ecg2_gamma[2],
                'ecg2_gamma_median': time_ecg2_gamma[3], 'ecg2_gamma_mod': time_ecg2_gamma[4], 'ecg2_gamma_var': time_ecg2_gamma[5],
                'ecg2_gamma_std': time_ecg2_gamma[6], 'ecg2_gamma_pow': time_ecg2_gamma[7], 'ecg2_gamma_tasa': time_ecg2_gamma[8],
                'ecg2_gamma_curt': time_ecg2_gamma[9], 'ecg2_gamma_ske': time_ecg2_gamma[10],
                
                'ecg2_gamma_max_freq': freq_ecg2_gamma[0], 'ecg2_gamma_min_freq': freq_ecg2_gamma[1], 'ecg2_gamma_med_freq': freq_ecg2_gamma[2], 
                'ecg2_gamma_median_freq': freq_ecg2_gamma[3], 'ecg2_gamma_mod_freq': freq_ecg2_gamma[4], 'ecg2_gamma_var_freq': freq_ecg2_gamma[5], 
                'ecg2_gamma_std_freq': freq_ecg2_gamma[6], 'ecg2_gamma_pow_freq': freq_ecg2_gamma[7], 'ecg2_gamma_tasa_freq': freq_ecg2_gamma[8], 
                'ecg2_gamma_curt_freq': freq_ecg2_gamma[9], 'ecg2_gamma_ske_freq': freq_ecg2_gamma[10], 'ecg2_gamma_freq_max': freq_ecg2_gamma[11], 
                'ecg2_gamma_mean_freq': freq_ecg2_gamma[12], 'ecg2_gamma_freq_median': freq_ecg2_gamma[13], 'ecg2_gamma_freq_centroid': freq_ecg2_gamma[14], 
                'ecg2_gamma_freq_dispersion': freq_ecg2_gamma[15], 'ecg2_gamma_freq_flatness': freq_ecg2_gamma[16], 'ecg2_gamma_freq_slope': freq_ecg2_gamma[17], 
                'ecg2_gamma_freq_crest_factor': freq_ecg2_gamma[18],
                            
            
                #EEG entropias
                #EEG1
                'eeg1_entropy_shan':ent_eeg1[0], 'eeg1_entropy_ren':ent_eeg1[1], 'eeg1_entropy_ApEn':ent_eeg1[2],'eeg1_entropy_Samp':ent_eeg1[3],
                'eeg1_spect_ent_f':ent_eeg1[4], 'eeg1_svd_ent':ent_eeg1[5],
                
                'eeg1_lenta_entropy_shan':ent_eeg1_lenta[0], 'eeg1_lenta_entropy_ren':ent_eeg1_lenta[1], 'eeg1_lenta_entropy_ApEn':ent_eeg1_lenta[2],'eeg1_lenta_entropy_Samp':ent_eeg1_lenta[3],
                'eeg1_lenta_spect_ent_f':ent_eeg1_lenta[4], 'eeg1_lenta_svd_ent':ent_eeg1_lenta[5],
                
                'eeg1_delta_entropy_shan': ent_eeg1_delta[0], 'eeg1_delta_entropy_ren': ent_eeg1_delta[1], 'eeg1_delta_entropy_ApEn': ent_eeg1_delta[2],'eeg1_delta_entropy_Samp': ent_eeg1_delta[3],
                'eeg1_delta_spect_ent_f': ent_eeg1_delta[4], 'eeg1_delta_svd_ent': ent_eeg1_delta[5],
                
                'eeg1_theta_entropy_shan': ent_eeg1_theta[0], 'eeg1_theta_entropy_ren': ent_eeg1_theta[1], 'eeg1_theta_entropy_ApEn': ent_eeg1_theta[2],'eeg1_theta_entropy_Samp': ent_eeg1_theta[3],
                'eeg1_theta_spect_ent_f': ent_eeg1_theta[4], 'eeg1_theta_svd_ent': ent_eeg1_theta[5],
                        
                'eeg1_alpha_entropy_shan': ent_eeg1_alpha[0], 'eeg1_alpha_entropy_ren': ent_eeg1_alpha[1], 'eeg1_alpha_entropy_ApEn': ent_eeg1_alpha[2],
                'eeg1_alpha_entropy_Samp': ent_eeg1_alpha[3], 'eeg1_alpha_spect_ent_f': ent_eeg1_alpha[4], 'eeg1_alpha_svd_ent': ent_eeg1_alpha[5],
            
                'eeg1_beta_entropy_shan': ent_eeg1_beta[0], 'eeg1_beta_entropy_ren': ent_eeg1_beta[1], 'eeg1_beta_entropy_ApEn': ent_eeg1_beta[2],
                'eeg1_beta_entropy_Samp': ent_eeg1_beta[3], 'eeg1_beta_spect_ent_f': ent_eeg1_beta[4], 'eeg1_beta_svd_ent': ent_eeg1_beta[5],
            
                'eeg1_gamma_entropy_shan': ent_eeg1_gamma[0], 'eeg1_gamma_entropy_ren': ent_eeg1_gamma[1], 'eeg1_gamma_entropy_ApEn': ent_eeg1_gamma[2],
                'eeg1_gamma_entropy_Samp': ent_eeg1_gamma[3], 'eeg1_gamma_spect_ent_f': ent_eeg1_gamma[4], 'eeg1_gamma_svd_ent': ent_eeg1_gamma[5],
                
            
                #EEG2
                'eeg2_entropy_shan':ent_eeg2[0], 'eeg2_entropy_ren':ent_eeg2[1], 'eeg2_entropy_ApEn':ent_eeg2[2],'eeg2_entropy_Samp':ent_eeg2[3],
                'eeg2_spect_ent_f':ent_eeg2[4], 'eeg2_svd_ent':ent_eeg2[5],
                
                'eeg2_lenta_entropy_shan': ent_eeg2_lenta[0], 'eeg2_lenta_entropy_ren': ent_eeg2_lenta[1], 'eeg2_lenta_entropy_ApEn': ent_eeg2_lenta[2], 'eeg2_lenta_entropy_Samp': ent_eeg2_lenta[3],
                'eeg2_lenta_spect_ent_f': ent_eeg2_lenta[4], 'eeg2_lenta_svd_ent': ent_eeg2_lenta[5], 
            
                'eeg2_delta_entropy_shan': ent_eeg2_delta[0], 'eeg2_delta_entropy_ren': ent_eeg2_delta[1], 'eeg2_delta_entropy_ApEn': ent_eeg2_delta[2], 'eeg2_delta_entropy_Samp': ent_eeg2_delta[3],
                'eeg2_delta_spect_ent_f': ent_eeg2_delta[4], 'eeg2_delta_svd_ent': ent_eeg2_delta[5], 
            
                'eeg2_theta_entropy_shan': ent_eeg2_theta[0], 'eeg2_theta_entropy_ren': ent_eeg2_theta[1], 'eeg2_theta_entropy_ApEn': ent_eeg2_theta[2], 'eeg2_theta_entropy_Samp': ent_eeg2_theta[3],
                'eeg2_theta_spect_ent_f': ent_eeg2_theta[4], 'eeg2_theta_svd_ent': ent_eeg2_theta[5], 
            
                'eeg2_alpha_entropy_shan': ent_eeg2_alpha[0], 'eeg2_alpha_entropy_ren': ent_eeg2_alpha[1], 'eeg2_alpha_entropy_ApEn': ent_eeg2_alpha[2], 'eeg2_alpha_entropy_Samp': ent_eeg2_alpha[3],
                'eeg2_alpha_spect_ent_f': ent_eeg2_alpha[4], 'eeg2_alpha_svd_ent': ent_eeg2_alpha[5], 
            
                'eeg2_beta_entropy_shan': ent_eeg2_beta[0], 'eeg2_beta_entropy_ren': ent_eeg2_beta[1], 'eeg2_beta_entropy_ApEn': ent_eeg2_beta[2], 'eeg2_beta_entropy_Samp': ent_eeg2_beta[3],
                'eeg2_beta_spect_ent_f': ent_eeg2_beta[4], 'eeg2_beta_svd_ent': ent_eeg2_beta[5], 
            
                'eeg2_gamma_entropy_shan': ent_eeg2_gamma[0], 'eeg2_gamma_entropy_ren': ent_eeg2_gamma[1], 'eeg2_gamma_entropy_ApEn': ent_eeg2_gamma[2], 'eeg2_gamma_entropy_Samp': ent_eeg2_gamma[3],
                'eeg2_gamma_spect_ent_f': ent_eeg2_gamma[4], 'eeg2_gamma_svd_ent': ent_eeg2_gamma[5], 
            
            
                #ECG entropias
                'ecg2_entropy_shan':ent_ecg2[0], 'ecg2_entropy_ren':ent_ecg2[1], 'ecg2_entropy_ApEn':ent_ecg2[2],'ecg2_entropy_Samp':ent_ecg2[3],
                'ecg2_spect_ent_f':ent_ecg2[4], 'ecg2_svd_ent':ent_ecg2[5],
                
                'ecg2_lenta_entropy_shan': ent_ecg2_lenta[0], 'ecg2_lenta_entropy_ren': ent_ecg2_lenta[1], 'ecg2_lenta_entropy_ApEn': ent_ecg2_lenta[2], 'ecg2_lenta_entropy_Samp': ent_ecg2_lenta[3],
                'ecg2_lenta_spect_ent_f': ent_ecg2_lenta[4], 'ecg2_lenta_svd_ent': ent_ecg2_lenta[5], 
            
                'ecg2_delta_entropy_shan': ent_ecg2_delta[0], 'ecg2_delta_entropy_ren': ent_ecg2_delta[1], 'ecg2_delta_entropy_ApEn': ent_ecg2_delta[2], 'ecg2_delta_entropy_Samp': ent_ecg2_delta[3],
                'ecg2_delta_spect_ent_f': ent_ecg2_delta[4], 'ecg2_delta_svd_ent': ent_ecg2_delta[5], 
                
                'ecg2_theta_entropy_shan': ent_ecg2_theta[0], 'ecg2_theta_entropy_ren': ent_ecg2_theta[1], 'ecg2_theta_entropy_ApEn': ent_ecg2_theta[2], 'ecg2_theta_entropy_Samp': ent_ecg2_theta[3],
                'ecg2_theta_spect_ent_f': ent_ecg2_theta[4], 'ecg2_theta_svd_ent': ent_ecg2_theta[5], 
            
                'ecg2_alpha_entropy_shan': ent_ecg2_alpha[0], 'ecg2_alpha_entropy_ren': ent_ecg2_alpha[1], 'ecg2_alpha_entropy_ApEn': ent_ecg2_alpha[2], 'ecg2_alpha_entropy_Samp': ent_ecg2_alpha[3],
                'ecg2_alpha_spect_ent_f': ent_ecg2_alpha[4], 'ecg2_alpha_svd_ent': ent_ecg2_alpha[5], 
                
                'ecg2_beta_entropy_shan': ent_ecg2_beta[0], 'ecg2_beta_entropy_ren': ent_ecg2_beta[1], 'ecg2_beta_entropy_ApEn': ent_ecg2_beta[2], 'ecg2_beta_entropy_Samp': ent_ecg2_beta[3],
                'ecg2_beta_spect_ent_f': ent_ecg2_beta[4], 'ecg2_beta_svd_ent': ent_ecg2_beta[5], 
                
                'ecg2_gamma_entropy_shan': ent_ecg2_gamma[0], 'ecg2_gamma_entropy_ren': ent_ecg2_gamma[1], 'ecg2_gamma_entropy_ApEn': ent_ecg2_gamma[2], 'ecg2_gamma_entropy_Samp': ent_ecg2_gamma[3],
                'ecg2_gamma_spect_ent_f': ent_ecg2_gamma[4], 'ecg2_gamma_svd_ent': ent_ecg2_gamma[5], 
                
                #EEG Caoticas
                #EEG1
                'eeg1_pfd':cao_eeg1[0], 'eeg1_h':cao_eeg1[1],'eeg1_complexity':cao_eeg1[2], 'eeg1_activity':cao_eeg1[3],'eeg1_mobility':cao_eeg1[4],'eeg1_katz':cao_eeg1[5], 'eeg1_higu_10':cao_eeg1[6], 'eeg1_higu_50':cao_eeg1[7], 'eeg1_higu_100':cao_eeg1[8],
                'eeg1_lenta_pfd':cao_eeg1_lenta[0],'eeg1_lenta_h':cao_eeg1_lenta[1], 'eeg1_lenta_complexity':cao_eeg1_lenta[2], 'eeg1_lenta_activity':cao_eeg1_lenta[3],'eeg1_lenta_mobility':cao_eeg1_lenta[4],'eeg1_lenta_katz':cao_eeg1_lenta[5], 'eeg1_lenta_higu_10':cao_eeg1_lenta[6], 'eeg1_lenta_higu_50':cao_eeg1_lenta[7], 'eeg1_lenta_higu_100':cao_eeg1_lenta[8],
                'eeg1_delta_pfd': cao_eeg1_delta[0], 'eeg1_delta_h': cao_eeg1_delta[1], 'eeg1_delta_complexity': cao_eeg1_delta[2], 'eeg1_delta_activity': cao_eeg1_delta[3], 'eeg1_delta_mobility': cao_eeg1_delta[4],'eeg1_delta_katz':cao_eeg1_delta[5], 'eeg1_delta_higu_10':cao_eeg1_delta[6], 'eeg1_delta_higu_50':cao_eeg1_delta[7], 'eeg1_delta_higu_100':cao_eeg1_delta[8],
                'eeg1_theta_pfd': cao_eeg1_theta[0], 'eeg1_theta_h': cao_eeg1_theta[1], 'eeg1_theta_complexity': cao_eeg1_theta[2], 'eeg1_theta_activity': cao_eeg1_theta[3], 'eeg1_theta_mobility': cao_eeg1_theta[4],'eeg1_theta_katz':cao_eeg1_theta[5], 'eeg1_theta_higu_10':cao_eeg1_theta[6], 'eeg1_theta_higu_50':cao_eeg1_theta[7], 'eeg1_theta_higu_100':cao_eeg1_theta[8],
                'eeg1_alpha_pfd': cao_eeg1_alpha[0], 'eeg1_alpha_h': cao_eeg1_alpha[1], 'eeg1_alpha_complexity': cao_eeg1_alpha[2], 'eeg1_alpha_activity': cao_eeg1_alpha[3], 'eeg1_alpha_mobility': cao_eeg1_alpha[4],'eeg1_alpha_katz':cao_eeg1_alpha[5], 'eeg1_alpha_higu_10':cao_eeg1_alpha[6], 'eeg1_alpha_higu_50':cao_eeg1_alpha[7], 'eeg1_alpha_higu_100':cao_eeg1_alpha[8],
                'eeg1_beta_pfd': cao_eeg1_beta[0], 'eeg1_beta_h': cao_eeg1_beta[1], 'eeg1_beta_complexity': cao_eeg1_beta[2], 'eeg1_beta_activity': cao_eeg1_beta[3], 'eeg1_beta_mobility': cao_eeg1_beta[4],'eeg1_beta_katz':cao_eeg1_beta[5], 'eeg1_beta_higu_10':cao_eeg1_beta[6], 'eeg1_beta_higu_50':cao_eeg1_beta[7], 'eeg1_beta_higu_100':cao_eeg1_beta[8],
                'eeg1_gamma_pfd': cao_eeg1_gamma[0], 'eeg1_gamma_h': cao_eeg1_gamma[1], 'eeg1_gamma_complexity': cao_eeg1_gamma[2], 'eeg1_gamma_activity': cao_eeg1_gamma[3], 'eeg1_gamma_mobility': cao_eeg1_gamma[4],'eeg1_gamma_katz':cao_eeg1_gamma[5], 'eeg1_gamma_higu_10':cao_eeg1_gamma[6], 'eeg1_gamma_higu_50':cao_eeg1_gamma[7], 'eeg1_gamma_higu_100':cao_eeg1_gamma[8],

            
                #EEG2
                'eeg2_pfd':cao_eeg2[0], 'eeg2_h':cao_eeg2[1],'eeg2_complexity':cao_eeg2[2], 'eeg2_activity':cao_eeg2[3],'eeg2_mobility':cao_eeg2[4],'eeg2_katz':cao_eeg2[5], 'eeg2_higu_10':cao_eeg2[6], 'eeg2_higu_50':cao_eeg2[7], 'eeg2_higu_100':cao_eeg2[8],
                'eeg2_lenta_pfd':cao_eeg2_lenta[0],'eeg2_lenta_h':cao_eeg2_lenta[1], 'eeg2_lenta_complexity':cao_eeg2_lenta[2], 'eeg2_lenta_activity':cao_eeg2_lenta[3],'eeg2_lenta_mobility':cao_eeg2_lenta[4],'eeg2_lenta_katz':cao_eeg2_lenta[5], 'eeg2_lenta_higu_10':cao_eeg2_lenta[6], 'eeg2_lenta_higu_50':cao_eeg2_lenta[7], 'eeg2_lenta_higu_100':cao_eeg2_lenta[8],
                'eeg2_delta_pfd': cao_eeg2_delta[0], 'eeg2_delta_h': cao_eeg2_delta[1], 'eeg2_delta_complexity': cao_eeg2_delta[2], 'eeg2_delta_activity': cao_eeg2_delta[3], 'eeg2_delta_mobility': cao_eeg2_delta[4],'eeg2_delta_katz':cao_eeg2_delta[5], 'eeg2_delta_higu_10':cao_eeg2_delta[6], 'eeg2_delta_higu_50':cao_eeg2_delta[7], 'eeg2_delta_higu_100':cao_eeg2_delta[8],
                'eeg2_theta_pfd': cao_eeg2_theta[0], 'eeg2_theta_h': cao_eeg2_theta[1], 'eeg2_theta_complexity': cao_eeg2_theta[2], 'eeg2_theta_activity': cao_eeg2_theta[3], 'eeg2_theta_mobility': cao_eeg2_theta[4],'eeg2_theta_katz':cao_eeg2_theta[5], 'eeg2_theta_higu_10':cao_eeg2_theta[6], 'eeg2_theta_higu_50':cao_eeg2_theta[7], 'eeg2_theta_higu_100':cao_eeg2_theta[8],
                'eeg2_alpha_pfd': cao_eeg2_alpha[0], 'eeg2_alpha_h': cao_eeg2_alpha[1], 'eeg2_alpha_complexity': cao_eeg2_alpha[2], 'eeg2_alpha_activity': cao_eeg2_alpha[3], 'eeg2_alpha_mobility': cao_eeg2_alpha[4],'eeg2_alpha_katz':cao_eeg2_alpha[5], 'eeg2_alpha_higu_10':cao_eeg2_alpha[6], 'eeg2_alpha_higu_50':cao_eeg2_alpha[7], 'eeg2_alpha_higu_100':cao_eeg2_alpha[8],
                'eeg2_beta_pfd': cao_eeg2_beta[0], 'eeg2_beta_h': cao_eeg2_beta[1], 'eeg2_beta_complexity': cao_eeg2_beta[2], 'eeg2_beta_activity': cao_eeg2_beta[3], 'eeg2_beta_mobility': cao_eeg2_beta[4],'eeg2_beta_katz':cao_eeg2_beta[5], 'eeg2_beta_higu_10':cao_eeg2_beta[6], 'eeg2_beta_higu_50':cao_eeg2_beta[7], 'eeg2_beta_higu_100':cao_eeg2_beta[8],
                'eeg2_gamma_pfd': cao_eeg2_gamma[0], 'eeg2_gamma_h': cao_eeg2_gamma[1], 'eeg2_gamma_complexity': cao_eeg2_gamma[2], 'eeg2_gamma_activity': cao_eeg2_gamma[3], 'eeg2_gamma_mobility': cao_eeg2_gamma[4],'eeg2_gamma_katz':cao_eeg2_gamma[5], 'eeg2_gamma_higu_10':cao_eeg2_gamma[6], 'eeg2_gamma_higu_50':cao_eeg2_gamma[7], 'eeg2_gamma_higu_100':cao_eeg2_gamma[8],

                
                #ECG Caoticas
                'ecg2_pfd':cao_ecg2[0], 'ecg2_h':cao_ecg2[1],'ecg2_complexity':cao_ecg2[2], 'ecg2_activity':cao_ecg2[3],'ecg2_mobility':cao_ecg2[4],'ecg2_katz':cao_ecg2[5], 'ecg2_higu_10':cao_ecg2[6], 'ecg2_higu_50':cao_ecg2[7], 'ecg2_higu_100':cao_ecg2[8],
                'ecg2_lenta_pfd':cao_ecg2_lenta[0],'ecg2_lenta_h':cao_ecg2_lenta[1], 'ecg2_lenta_complexity':cao_ecg2_lenta[2], 'ecg2_lenta_activity':cao_ecg2_lenta[3],'ecg2_lenta_mobility':cao_ecg2_lenta[4],'ecg2_lenta_katz':cao_ecg2_lenta[5], 'ecg2_lenta_higu_10':cao_ecg2_lenta[6], 'ecg2_lenta_higu_50':cao_ecg2_lenta[7], 'ecg2_lenta_higu_100':cao_ecg2_lenta[8],
                'ecg2_delta_pfd': cao_ecg2_delta[0], 'ecg2_delta_h': cao_ecg2_delta[1], 'ecg2_delta_complexity': cao_ecg2_delta[2], 'ecg2_delta_activity': cao_ecg2_delta[3], 'ecg2_delta_mobility': cao_ecg2_delta[4],'ecg2_delta_katz':cao_ecg2_delta[5], 'ecg2_delta_higu_10':cao_ecg2_delta[6], 'ecg2_delta_higu_50':cao_ecg2_delta[7], 'ecg2_delta_higu_100':cao_ecg2_delta[8],
                'ecg2_theta_pfd': cao_ecg2_theta[0], 'ecg2_theta_h': cao_ecg2_theta[1], 'ecg2_theta_complexity': cao_ecg2_theta[2], 'ecg2_theta_activity': cao_ecg2_theta[3], 'ecg2_theta_mobility': cao_ecg2_theta[4],'ecg2_theta_katz':cao_ecg2_theta[5], 'ecg2_theta_higu_10':cao_ecg2_theta[6], 'ecg2_theta_higu_50':cao_ecg2_theta[7], 'ecg2_theta_higu_100':cao_ecg2_theta[8],
                'ecg2_alpha_pfd': cao_ecg2_alpha[0], 'ecg2_alpha_h': cao_ecg2_alpha[1], 'ecg2_alpha_complexity': cao_ecg2_alpha[2], 'ecg2_alpha_activity': cao_ecg2_alpha[3], 'ecg2_alpha_mobility': cao_ecg2_alpha[4],'ecg2_alpha_katz':cao_ecg2_alpha[5], 'ecg2_alpha_higu_10':cao_ecg2_alpha[6], 'ecg2_alpha_higu_50':cao_ecg2_alpha[7], 'ecg2_alpha_higu_100':cao_ecg2_alpha[8],
                'ecg2_beta_pfd': cao_ecg2_beta[0], 'ecg2_beta_h': cao_ecg2_beta[1], 'ecg2_beta_complexity': cao_ecg2_beta[2], 'ecg2_beta_activity': cao_ecg2_beta[3], 'ecg2_beta_mobility': cao_ecg2_beta[4],'ecg2_beta_katz':cao_ecg2_beta[5], 'ecg2_beta_higu_10':cao_ecg2_beta[6], 'ecg2_beta_higu_50':cao_ecg2_beta[7], 'ecg2_beta_higu_100':cao_ecg2_beta[8],
                'ecg2_gamma_pfd': cao_ecg2_gamma[0], 'ecg2_gamma_h': cao_ecg2_gamma[1], 'ecg2_gamma_complexity': cao_ecg2_gamma[2], 'ecg2_gamma_activity': cao_ecg2_gamma[3], 'ecg2_gamma_mobility': cao_ecg2_gamma[4],'ecg2_gamma_katz':cao_ecg2_gamma[5], 'ecg2_gamma_higu_10':cao_ecg2_gamma[6], 'ecg2_gamma_higu_50':cao_ecg2_gamma[7], 'ecg2_gamma_higu_100':cao_ecg2_gamma[8],
                            
            
                # HRV dominio del tiempo
                'hrv_mean_nni': time_hrv[0],'hrv_sdnn': time_hrv[1],'hrv_sdsd': time_hrv[2],'hrv_nni_50': time_hrv[3],'hrv_pnni_50': time_hrv[4],
                'hrv_nni_20': time_hrv[5],'hrv_pnni_20': time_hrv[6],'hrv_rmssd': time_hrv[7],'hrv_median_nni': time_hrv[8],'hrv_range_nni': time_hrv[9],
                'hrv_cvsd': time_hrv[10],'hrv_cvnni': time_hrv[11],'hrv_mean_hr': time_hrv[12],'hrv_max_hr': time_hrv[13],'hrv_min_hr': time_hrv[14], 'hrv_std_hr': time_hrv[15],
            
                #HRV dominio de la frecuencia
                'hrv_lf':frec_hrv[0], 'hrv_hf':frec_hrv[1], 'hrv_lf_hf_ratio':frec_hrv[2], 'hrv_lfnu':frec_hrv[3], 'hrv_hfnu':frec_hrv[4],
                'hrv_total_power':frec_hrv[5], 'hrv_vlf':frec_hrv[6], 'hrv_tri_idx':frec_hrv[7],'hrv_sd1':frec_hrv[8],
                'hrv_sd2':frec_hrv[9],'hrv_ratio_sd2_sd1':frec_hrv[10] ,'hrv_csi':frec_hrv[11], 'hrv_cvi':frec_hrv[12],'hrv_Modified_csi':frec_hrv[13], 'hrv_elipse':frec_hrv[14],
            
                #Caracteristicas hrv, hf y lf
                # Tiempo
                'hf_max':time_hf[0], 'hf_min':time_hf[1], 'hf_med':time_hf[2], 'hf_median':time_hf[3], 'hf_mod':time_hf[4], 'hf_var':time_hf[5],
                'hf_std':time_hf[6], 'hf_pow':time_hf[7] , 'hf_tasa':time_hf[8], 'hf_curt':time_hf[9],'hf_ske':time_hf[10],
            
                'lf_max': time_lf[0], 'lf_min': time_lf[1], 'lf_med': time_lf[2], 'lf_median': time_lf[3], 'lf_mod': time_lf[4], 'lf_var': time_lf[5],
                'lf_std': time_lf[6], 'lf_pow': time_lf[7], 'lf_tasa': time_lf[8], 'lf_curt': time_lf[9], 'lf_ske': time_lf[10],
            
                # Entropia
                'hf_entropy_shan': ent_hf[0], 'hf_entropy_ren': ent_hf[1], 'hf_entropy_ApEn': ent_hf[2], 'hf_entropy_Samp': ent_hf[3],
                'hf_spect_ent_f': ent_hf[4], 'hf_svd_ent': ent_hf[5],
            
                'lf_entropy_shan': ent_lf[0], 'lf_entropy_ren': ent_lf[1], 'lf_entropy_ApEn': ent_lf[2], 'lf_entropy_Samp': ent_lf[3],
                'lf_spect_ent_f': ent_lf[4], 'lf_svd_ent': ent_lf[5], 
            
                'hrv_entropy_shan': ent_hrv[0], 'hrv_entropy_ren': ent_hrv[1], 'hrv_entropy_ApEn': ent_hrv[2], 'hrv_entropy_Samp': ent_hrv[3],
                'hrv_spect_ent_f': ent_hrv[4], 'hrv_svd_ent': ent_hrv[5], 
                
                # Caoticas
                'hf_pfd': cao_hf[0], 'hf_h': cao_hf[1], 'hf_complexity': cao_hf[2], 'hf_activity': cao_hf[3],'hf_mobility': cao_hf[4],'hf_katz':cao_hf[5], 'hf_higu_10':cao_hf[6], 'hf_higu_50':cao_hf[7], 'hf_higu_100':cao_hf[8], 'hf_dfa_a1':cao_hf[9], 'hf_dfa_a2':cao_hf[10],
                'lf_pfd': cao_lf[0], 'lf_h': cao_lf[1],'lf_complexity': cao_lf[2], 'lf_activity': cao_lf[3],'lf_mobility': cao_lf[4],'lf_katz':cao_lf[5], 'lf_higu_10':cao_lf[6], 'lf_higu_50':cao_lf[7], 'lf_higu_100':cao_lf[8], 'lf_dfa_a1':cao_lf[9], 'lf_dfa_a2':cao_lf[10],
                'hrv_pfd': cao_hrv[0], 'hrv_h': cao_hrv[1],'hrv_complexity': cao_hrv[2], 'hrv_activity': cao_hrv[3], 'hrv_mobility': cao_hrv[4], 'hrv_katz':cao_hrv[5], 'hrv_higu_10':cao_hrv[6], 'hrv_higu_50':cao_hrv[7], 'hrv_higu_100':cao_hrv[8], 'hrv_dfa_a1':cao_hrv[9], 'hrv_dfa_a2':cao_hrv[10],

                
                # Band power
                'Pabs_eeg1_lenta': bp_eeg1_lenta[0],'Prel_eeg1_lenta': bp_eeg1_lenta[1],
                'Pabs_eeg1_delta': bp_eeg1_delta[0], 'Prel_eeg1_delta': bp_eeg1_delta[1],
                'Pabs_eeg1_theta': bp_eeg1_theta[0], 'Prel_eeg1_theta': bp_eeg1_theta[1],
                'Pabs_eeg1_alpha': bp_eeg1_alpha[0], 'Prel_eeg1_alpha': bp_eeg1_alpha[1],
                'Pabs_eeg1_beta': bp_eeg1_beta[0], 'Prel_eeg1_beta': bp_eeg1_beta[1],
                'Pabs_eeg1_gamma': bp_eeg1_gamma[0], 'Prel_eeg1_gamma': bp_eeg1_gamma[1],
                
                'Pabs_eeg2_lenta': bp_eeg2_lenta[0],'Prel_eeg2_lenta': bp_eeg2_lenta[1],
                'Pabs_eeg2_delta': bp_eeg2_delta[0], 'Prel_eeg2_delta': bp_eeg2_delta[1],
                'Pabs_eeg2_theta': bp_eeg2_theta[0], 'Prel_eeg2_theta': bp_eeg2_theta[1],
                'Pabs_eeg2_alpha': bp_eeg2_alpha[0], 'Prel_eeg2_alpha': bp_eeg2_alpha[1],
                'Pabs_eeg2_beta': bp_eeg2_beta[0], 'Prel_eeg2_beta': bp_eeg2_beta[1],
                'Pabs_eeg2_gamma': bp_eeg2_gamma[0], 'Prel_eeg2_gamma': bp_eeg2_gamma[1],
                
                'Pabs_ecg2_lenta': bp_ecg2_lenta[0],'Prel_ecg2_lenta': bp_ecg2_lenta[1],
                'Pabs_ecg2_delta': bp_ecg2_delta[0], 'Prel_ecg2_delta': bp_ecg2_delta[1],
                'Pabs_ecg2_theta': bp_ecg2_theta[0], 'Prel_ecg2_theta': bp_ecg2_theta[1],
                'Pabs_ecg2_alpha': bp_ecg2_alpha[0], 'Prel_ecg2_alpha': bp_ecg2_alpha[1],
                'Pabs_ecg2_beta': bp_ecg2_beta[0], 'Prel_ecg2_beta': bp_ecg2_beta[1],
                'Pabs_ecg2_gamma': bp_ecg2_gamma[0], 'Prel_ecg2_gamma': bp_ecg2_gamma[1],
                
                #RMS
                'eeg1_lenta_rms': rms_eeg1_lenta, 'eeg1_delta_rms': rms_eeg1_delta, 'eeg1_theta_rms': rms_eeg1_theta,
                'eeg1_alpha_rms': rms_eeg1_alpha, 'eeg1_beta_rms': rms_eeg1_beta,  'eeg1_gamma_rms': rms_eeg1_gamma, 
                
                'eeg2_lenta_rms': rms_eeg2_lenta, 'eeg2_delta_rms': rms_eeg2_delta, 'eeg2_theta_rms': rms_eeg2_theta,
                'eeg2_alpha_rms': rms_eeg2_alpha, 'eeg2_beta_rms': rms_eeg2_beta, 'eeg2_gamma_rms': rms_eeg2_gamma,
                
                'ecg2_lenta_rms': rms_ecg2_lenta, 'ecg2_delta_rms': rms_ecg2_delta, 'ecg2_theta_rms': rms_ecg2_theta,
                'ecg2_alpha_rms': rms_ecg2_alpha, 'ecg2_beta_rms': rms_ecg2_beta, 'ecg2_gamma_rms': rms_ecg2_gamma,
            
                #RMS espectros
                'esp_rms_eeg1':esp_rms_eeg1,
                'esp_rms_eeg1_lenta':esp_rms_eeg1_lenta, 'esp_rms_eeg1_delta':esp_rms_eeg1_delta,'esp_rms_eeg1_theta':esp_rms_eeg1_theta,
                'esp_rms_eeg1_alpha': esp_rms_eeg1_alpha , 'esp_rms_eeg1_beta':esp_rms_eeg1_beta, 'esp_rms_eeg1_gamma':esp_rms_eeg1_gamma, 
                
                'esp_rms_eeg2':esp_rms_eeg2,
                'esp_rms_eeg2_lenta': esp_rms_eeg2_lenta, 'esp_rms_eeg2_delta': esp_rms_eeg2_delta, 'esp_rms_eeg2_theta': esp_rms_eeg2_theta,
                'esp_rms_eeg2_alpha': esp_rms_eeg2_alpha, 'esp_rms_eeg2_beta': esp_rms_eeg2_beta, 'esp_rms_eeg2_gamma': esp_rms_eeg2_gamma,
                 
                'esp_rms_ecg2':esp_rms_ecg2,
                'esp_rms_ecg2_lenta': esp_rms_ecg2_lenta, 'esp_rms_ecg2_delta': esp_rms_ecg2_delta,'esp_rms_ecg2_theta': esp_rms_ecg2_theta,
                'esp_rms_ecg2_alpha': esp_rms_ecg2_alpha , 'esp_rms_ecg2_beta': esp_rms_ecg2_beta, 'esp_rms_ecg2_gamma': esp_rms_ecg2_gamma,

           
            
            })
    
            # Agregar el diccionario a la lista
            data_list.append(current_data_dict)
            
            
        df_dataset = pd.DataFrame(data_list)
              
        # Saving the dataframe in a file
        if (caseid_indx == startindex): 
            name_str = "/hpc/gpfs2/scratch/u/idroboen/dataset/sofia/filters_60/" + "df_dataset_filt_" + str(startindex) + '_' + str(stopindex) + ".csv" 
            filepath = Path(name_str)  
            df_dataset.to_csv(filepath, encoding='utf-8', header=True, na_rep='NaN', float_format='%.6f')
            
        if (caseid_indx > startindex): 
            name_str = "/hpc/gpfs2/scratch/u/idroboen/dataset/sofia/filters_60/" + "df_dataset_filt_" + str(startindex) + '_' + str(stopindex) + ".csv" 
            filepath = Path(name_str)  
            df_dataset.to_csv(filepath, mode='a', encoding='utf-8', header=False, na_rep='NaN', float_format='%.6f')     # na_rep='NaN'
    

    
    
    
    
# %% Main
def main():
    parser = argparse.ArgumentParser(description='Run analysis')
    parser.add_argument('--start', metavar='ID', required=True,
                       help='start ID')
    parser.add_argument('--stop', metavar='ID', required=True,
                       help='stop ID')
    args = parser.parse_args()
    
    main_processing(startindex=int(args.start), stopindex=int(args.stop))

if __name__ == '__main__':
    main()