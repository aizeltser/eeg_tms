#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mne')


# In[56]:


import mne
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import seaborn as sns
import scipy.stats as stats
from IPython.display import display


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:


sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12


# In[60]:


results = []
#Конфигурация
root_folder = "eeg_env_recordings/"
evoked_folder = "processed_data/"
os.makedirs(evoked_folder, exist_ok = True)


# In[62]:


#Окна для анализа вызванных потенциалов
peak_windows = {'N1': (0.080, 0.120), 'P2': (0.140, 0.220), 'MMN': (0.160, 0.240), 'N2': (0.260, 0.340), 'P300': (0.300, 0.500), 'N400': (0.350, 0.600)}


# In[64]:


#Группировка каналов
frontal_channels = ['Fp1', 'Fp2']
temporal_channels = ['T3', 'T4']
posterior_channels = ['O1', 'O2']


# In[66]:


#Минимальный рабочий код для 1 файла
#file_path = "eeg_env_recordings/2СН7SRX/03.10_2СН7SRX_4.EDF"

#raw = mne.io.read_raw_edf(file_path, preload=True)
#raw.set_eeg_reference('average', projection=True)
#raw.set_montage('standard_1020')
#raw.filter(0.1, 30, fir_design="firwin")
#print(raw.info)

#ica = ICA(n_components=6, random_state=97, max_iter=800)
#ica.fit(raw)

#ica.plot_components()
#plt.show() 


# In[15]:


for dirpath, dirnames, filenames in os.walk(root_folder):
    for file in filenames:
        if file.endswith(".EDF"):
            file_path = os.path.join(dirpath, file)
            print(f"Идет обработка файла {file_path}")
            try:
                raw = mne.io.read_raw_edf(file_path, preload = True)
                raw.set_eeg_reference('average', projection = True)
                raw.set_montage('standard_1020')
                raw.filter(1, 30, fir_design = "firwin")
                print(raw.info)
            
                #ICA для удаления артефактов (моргание и саккады) - удаление и выбор компонент вручную
                # ica = mne.preprocessing.ICA(n_components = 6, random_state = 97, max_iter = 800)
                # ica.fit(raw)
                # ica.plot_components()
                # plt.show()
                # exclude_input = input("Введите номера ICA-компонентов через запятую, которые нужно исключить:")
                # if exclude_input.strip():
                #     ica.exclude = [int(x.strip()) for x in exclude_input.split(",")]
                # else:
                #     ica.exclude = []

                #ICA для удаления артефактов - автоматическое обнаружение с помощью Fp1 и Fp2 как EOG каналов
                eog_signal = raw.copy().pick_channels(['Fp1', 'Fp2']).get_data()
                eog_surrogate = eog_signal[0] - eog_signal[1]
                temp_info = mne.create_info(['EOG'], sfreq=raw.info['sfreq'], ch_types=['eog'])
                raw_eog = mne.io.RawArray(eog_surrogate[np.newaxis, :], temp_info)
                raw_with_eog = raw.copy()
                raw_with_eog.add_channels([raw_eog], force_update_info = True)
                
                ica = mne.preprocessing.ICA(n_components = 6, random_state = 97, max_iter = 800)
                ica.fit(raw_with_eog)
                eog_indices, eog_scores = ica.find_bads_eog(raw_with_eog, ch_name = 'EOG')
                ica.exclude = eog_indices
                if eog_indices:
                    print(f"Обнаружена артефакты в компонентах: {eog_indices}")
                    ica.plot_properties(raw_eog, picks = eog_indices)
                    plt.show()
                raw_ica = ica.apply(raw.copy())
                
                # exclude_input = input("Введите номера компонентов ICA через запятую, которые вы бы также хотели исключить или Enter, чтобы пропустить):")
                # if exclude_input.strip():
                #     ica.exclude.extend([int(x.strip()) for x in exclude_input.split(",") if x.strip().isdigit()])
                # raw_ica = ica.apply(raw.copy())
            
                #Делаем разметку по ударам метронома
                bpm = 120
                interval = 60 / bpm
                duration = raw.times[-1]
                sfreq = raw.info['sfreq']

                events = []
                for i in range(int(duration // interval)):
                    onset = i * interval
                    desc = 'special' if (i+1) % 8 == 0 else 'normal'
                    events.append([int(onset*sfreq), 0, 2 if desc == 'special' else 1])
                events = np.array(events)
                event_id = {'normal':1, 'special':2}

                #Эпохи
                epochs = mne.Epochs(raw_ica, events, event_id = event_id, tmin = -0.1, tmax = 0.6, baseline = (-0.1, 0), reject = dict(eeg=75e-6), flat = dict(eeg=0.5e-6), preload = True)
                subject_id = os.path.splitext(file)[0]
                output_dir = "processed_data"
                os.makedirs(output_dir, exist_ok = True)
                epochs.save(f"{output_dir}/{subject_id}-epo.fif", overwrite = True)

                #ERP
                evoked_norm = epochs['normal'].average()
                evoked_spec = epochs['special'].average()
                evoked_norm.save(f'{output_dir}/{subject_id}-normal-ave.fif', overwrite = True)
                evoked_spec.save(f'{output_dir}/{subject_id}-special-ave.fif', overwrite = True)
                fig_spec = evoked_spec.plot(spatial_colors = True, show = False)
                fig_spec.suptitle(f"Special tone ERP - {subject_id}")
                fig_norm = evoked_norm.plot(spatial_colors = True, show = False)
                fig_norm.suptitle(f"Normal tone ERP - {subject_id}")

                plt.show()

            except Exception as e:
                print(f"Возникла ошибка при обработке файла {file_path}: {str(e)}")
                continue

print("\nВсе файлы успешно обработаны!")


# In[18]:


for file in os.listdir(evoked_folder):
    if file.endswith('-normal-ave.fif') or file.endswith('-special-ave.fif'):
        
        basename = os.path.splitext(file)[0]
        date, rest = basename.split('_', 1)
        *middle_parts, record_part = rest.rsplit('_', 1)
        sub_id = '_'.join(middle_parts)
        record_num = record_part.split('-')[0]
        

        if None in (date, sub_id, record_num):
            print(f"Не удалось извлечь данные из имени файла: {file}")
            continue
        
        condition = 'normal' if 'normal' in file else 'special'
        filepath = os.path.join(evoked_folder, file)
        try:
            evoked_list = mne.read_evokeds(filepath)
            evoked = None
            for ev in evoked_list:
                if ('normal' in file and 'normal' in ev.comment.lower()) or \
                   ('special' in file and 'special' in ev.comment.lower()):
                    evoked = ev
                    break
            
            if evoked is None:
                print(f"Не удалось найти соответствующее условие в {file}")
                continue
            for comp, (tmin, tmax) in peak_windows.items():
                if comp in ['N1', 'P2', 'MMN', 'N2']:
                    area = 'frontal'
                elif comp in ['N400']:
                    area = 'temporal'
                else:
                    area = 'posterior'
                if area == 'frontal':
                    picks = [ch for ch in frontal_channels if ch in evoked.ch_names]
                elif area == 'temporal':
                    picks = [ch for ch in temporal_channels if ch in evoked.ch_names]
                else:
                    picks = [ch for ch in posterior_channels if ch in evoked.ch_names]

                if not picks:
                    print(f"Для компонента {comp} не найдены каналы в файле {file}")
                    continue

                evoked_avg = evoked.copy().pick(picks).data.mean(axis = 0)
                times = evoked.times
                idx_min = np.argmin(np.abs(times - tmin))
                idx_max = np.argmin(np.abs(times - tmax))

                segment = evoked_avg[idx_min:idx_max]
                if 'P' in comp:
                    peak_amp = segment.max()
                    peak_idx = segment.argmax()
                else:
                    peak_amp = segment.min()
                    peak_idx = segment.argmin()

                peak_latency = times[idx_min + peak_idx]

                results.append({'Date': date, 'Sub': sub_id, 'Recording number': record_num, 'Condition': condition, 'Component': comp, 'Region': area, 'Latency_seconds': peak_latency, 'Amplitude_uV': peak_amp * 1e6})
        except Exception as e:
            print(f"Ошибка при обработке файла {file}: {str(e)}")
            continue


# In[19]:


if results:
    df = pd.DataFrame(results)
    df['Session'] = df['Recording number'].apply(
        lambda x: 'Pre-TMS' if int(x) <= 5 else 'Post-TMS'
    )
    df['Group'] = df['Sub'].apply(lambda x: 'Control' if 'контроль' in x.lower() else 'TMS')
    df.sort_values(by = ['Session', 'Condition', 'Sub', 'Component'])
    df.to_csv('ERP_summary.csv', index = False)
    print(f"Сохранено {len(df)} записей в ERP_summary.csv")
    print(df.head())
else:
    print(f"Не найдено данных для сохранения.")


# In[90]:


get_ipython().system('pip install xlsxwriter')
from IPython.display import display
output_dir = "ERP_analysis_results"
os.makedirs(output_dir, exist_ok=True)
#Настройки отображения
pd.set_option('display.max_columns', None)
plt.style.use('default')
sns.set_theme(style='whitegrid', palette='Set2')

#Функция для подсветки значимых значений
def highlight_significant(val):
    color = 'lightgreen' if val < 0.05 else ''
    return f'background-color: {color}'

components = df['Component'].unique()
sessions = df['Session'].unique()
conditions = df['Condition'].unique()
stat_results = []

for comp in components:
    comp_df = df[df['Component'] == comp]
    
    for session in sessions:
        for condition in conditions:
            #Выбираем данные для контрольной группы
            control = comp_df[(comp_df['Group'] == 'Control') & 
                            (comp_df['Session'] == session) &
                            (comp_df['Condition'] == condition)]['Amplitude_uV']
            
            #Выбираем данные для TMS группы
            tms = comp_df[(comp_df['Group'] == 'TMS') & 
                        (comp_df['Session'] == session) &
                        (comp_df['Condition'] == condition)]['Amplitude_uV']
            
            #Проверка на минимальное количество наблюдений
            if len(control) < 3 or len(tms) < 3:
                print(f"Недостаточно данных для {comp}, {session}, {condition}")
                continue
            
            #Проверка нормальности распределения
            _, p_control = stats.shapiro(control)
            _, p_tms = stats.shapiro(tms)
            
            if p_control > 0.05 and p_tms > 0.05:
                #T-тест для нормальных распределений
                stat, p_val = stats.ttest_ind(control, tms, equal_var=False)
                test_name = 'T-test'
            else:
                #Тест Манна-Уитни для ненормальных распределений
                stat, p_val = stats.mannwhitneyu(control, tms, alternative='two-sided')
                test_name = 'Mann-Whitney U'
            
            stat_results.append({
                'Component': comp,
                'Session': session,
                'Condition': condition,
                'Test': test_name,
                'Statistic': stat,
                'p-value': p_val,
                'Control_mean': control.mean(),
                'TMS_mean': tms.mean(),
                'Control_std': control.std(),
                'TMS_std': tms.std(),
                'N_control': len(control),
                'N_TMS': len(tms)
            })

#Создаем DataFrame с результатами
stat_df = pd.DataFrame(stat_results)
stat_df = stat_df.sort_values(['Component', 'Session', 'Condition'])

#Отображаем результаты
styled_df = stat_df.style.map(highlight_significant, subset=['p-value'])
html_path = os.path.join(output_dir, "statistical_results.html")
with open(html_path, 'w') as f:
    f.write(styled_df.to_html())

try:
    excel_path = os.path.join(output_dir, "ERP_statistical_results.xlsx")
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        stat_df.to_excel(writer, sheet_name='Results', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Results']
        format_green = workbook.add_format({'bg_color': '#C6EFCE'})
        
        for row in range(1, len(stat_df)+1):
            p_val = stat_df.loc[row-1, 'p-value']
            if p_val < 0.05:
                worksheet.set_row(row, None, format_green)
    
    print(f"Статистические результаты сохранены в {excel_path}")
except Exception as e:
    csv_path = os.path.join(output_dir, "ERP_statistical_results.csv")
    stat_df.to_csv(csv_path, index=False)
    print(f"Статистические результаты сохранены в {csv_path} (ошибка Excel: {e})")

#Сохраняем графики
for comp in components:
    comp_dir = os.path.join(output_dir, f"Component_{comp}")
    os.makedirs(comp_dir, exist_ok=True)
    
    #Boxplot по группам и сессиям
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='Session', y='Amplitude_uV', hue='Group', 
               data=df[df['Component'] == comp], palette='Set2')
    plt.title(f'Амплитуда компонента {comp} по группам и сессиям')
    plt.ylabel('Амплитуда (μV)')
    plt.xlabel('Сессия')
    plt.legend(title='Группа')
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, f'boxplot_{comp}_groups_sessions.png'), dpi=300)
    plt.close()
    
    #График средних значений
    plt.figure(figsize=(14, 6))
    sns.pointplot(x='Session', y='Amplitude_uV', hue='Group', 
                 data=df[df['Component'] == comp], palette='Set2',
                 errorbar=('ci', 95), dodge=True)
    plt.title(f'Средняя амплитуда компонента {comp} с 95% доверительными интервалами')
    plt.ylabel('Амплитуда (μV)')
    plt.xlabel('Сессия')
    plt.legend(title='Группа')
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, f'pointplot_{comp}_means.png'), dpi=300)
    plt.close()
    
    #Графики по условиям
    for condition in ['normal', 'special']:
        cond_data = df[(df['Component'] == comp) & (df['Condition'] == condition)]
        
        if not cond_data.empty:
            plt.figure(figsize=(14, 6))
            sns.boxplot(x='Session', y='Amplitude_uV', hue='Group',
                       data=cond_data, palette='Set2')
            plt.title(f'Амплитуда компонента {comp} ({condition}) по группам и сессиям')
            plt.ylabel('Амплитуда (μV)')
            plt.xlabel('Сессия')
            plt.legend(title='Группа')
            plt.tight_layout()
            plt.savefig(os.path.join(comp_dir, f'boxplot_{comp}_{condition}.png'), dpi=300)
            plt.close()

    #Топографические карты
    if 'evoked_spec' in locals() and 'evoked_norm' in locals():
        try:
            from mne.viz import plot_topomap
            import mne
            
            # Создаем искусственные данные для топографии (пример)
            times = [peak_windows[comp][0] + (peak_windows[comp][1]-peak_windows[comp][0])/2]
            fig = evoked_spec.plot_topomap(times=times, show=False)
            fig.savefig(os.path.join(comp_dir, f'topomap_{comp}.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Ошибка при создании топографии для {comp}: {e}")
    
    #Графики временных рядов
    plt.figure(figsize=(14, 6))
    for group in ['Control', 'TMS']:
        group_data = df[(df['Component'] == comp) & (df['Group'] == group)]
        sns.lineplot(data=group_data, x='Latency_seconds', y='Amplitude_uV', 
                    label=group, errorbar='se')
    plt.title(f'ERP Waveform для {comp} по группам')
    plt.xlabel('Время (сек)')
    plt.ylabel('Амплитуда (μV)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, f'erp_waveform_{comp}.png'), dpi=300)
    plt.close()

#Матрицы латентностей и амплитуд (для всех компонентов)
matrix_dir = os.path.join(output_dir, "Matrices")
os.makedirs(matrix_dir, exist_ok=True)

#Матрица средних амплитуд
pivot_amp = df.pivot_table(values='Amplitude_uV', 
                          index='Component', 
                          columns='Group', 
                          aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_amp, annot=True, cmap='coolwarm', fmt=".1f")
plt.title('Средние амплитуды по компонентам и группам')
plt.tight_layout()
plt.savefig(os.path.join(matrix_dir, 'amplitude_matrix.png'), dpi=300)
plt.close()

#Матрица латентностей
pivot_lat = df.pivot_table(values='Latency_seconds', 
                          index='Component', 
                          columns='Group', 
                          aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_lat, annot=True, cmap='viridis', fmt=".3f")
plt.title('Средние латентности по компонентам и группам')
plt.tight_layout()
plt.savefig(os.path.join(matrix_dir, 'latency_matrix.png'), dpi=300)
plt.close()

#Сохраняем текстовый отчет
report_path = os.path.join(output_dir, "analysis_report.txt")
with open(report_path, 'w') as f:
    f.write(f"Проанализировано компонентов: {len(components)}\n")
    f.write(f"Всего сравнений: {len(stat_results)}\n")
    f.write("\nЗначимые различия (p < 0.05):\n")
    
    significant = stat_df[stat_df['p-value'] < 0.05]
    if len(significant) > 0:
        f.write(significant.to_string(index=False))
    else:
        f.write("Не найдено значимых различий между группами")

print(f"\nВсе результаты сохранены в папку: {os.path.abspath(output_dir)}")
print(f"Содержимое папки: {os.listdir(output_dir)}")





