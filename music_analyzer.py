import os
import sys
import pandas as pd
import numpy as np
from pydub import AudioSegment
import librosa
import librosa.display
import math

MUSIC_FOLDER = 'music_files'
OUTPUT_CSV_FILE = 'MusicAnalysisResults.csv'

SUPPORTED_EXTENSIONS = ('.mp3', '.wav', '.flac', '.ogg', '.aac')

VALENCE_CENTROIDS = {
    "밝은": {"SpectralCentroid_Mean": 4525.0553, "Spectral_Rolloff_Mean": 9716.6697}, # Major Key, 높은 스펙트럼 중심/롤오프
    "보통": {"SpectralCentroid_Mean": 3179.7568, "Spectral_Rolloff_Mean": 6639.5068}, # Major Key, 중간 스펙트럼 중심/롤오프
    "어두운": {"SpectralCentroid_Mean": 1834.4584, "Spectral_Rolloff_Mean": 3562.3439}, # Minor Key, 낮은 스펙트럼 중심/롤오프
}
AROUSAL_CENTROIDS = {
    "역동적인": {"BPM": 134.1619, "RMS_Mean": 0.3428}, # 높은 BPM, 높은 RMS
    "온화한": {"BPM": 123.0013, "RMS_Mean": 0.2}, # 중간 BPM, 중간 RMS
    "차분한": {"BPM": 111.8407, "RMS_Mean": 0.0572}, # 낮은 BPM, 낮은 RMS
}
INTENSITY_CENTROIDS = {
    "강렬한": {"RMS_Mean": 0.2590, "SpectralFlatness_Mean": 0.0298}, # 높은 RMS, 높은 평탄도
    "온건한": {"RMS_Mean": 0.2, "SpectralFlatness_Mean": 0.0171}, # 중간 RMS, 중간 평탄도
    "잔잔한": {"RMS_Mean": 0.1409, "SpectralFlatness_Mean": 0.0044}, # 낮은 RMS, 낮은 평탄도
}
COMPLEXITY_CENTROIDS = {
    "복잡한": {"RhythmComplexity_Score": 1.05, "SpectralFlatness_Mean": 0.022}, # 높은 리듬 복잡성, 높은 평탄도
    "규칙적인": {"RhythmComplexity_Score": 0.8, "SpectralFlatness_Mean": 0.015}, # 중간 리듬 복잡성, 중간 평탄도
    "단순한": {"RhythmComplexity_Score": 0.55, "SpectralFlatness_Mean": 0.08}, # 낮은 리듬 복잡성, 낮은 평탄도
}
KEY_SCORES = {
    'C Major': 7.66666667, 'C# Major': 6.66666667, 'D Major': 8.66666667,
    'D# Major': 6.33333333, 'E Major': 9.66666667, 'F Major': 6.66666667,
    'F# Major': 8.33333333, 'G Major': 8.66666667, 'G# Major': 6.33333333,
    'A Major': 10, 'A# Major': 7, 'B Major': 9,
    'C Minor': -7, 'C# Minor': -7.33333333, 'D Minor': -6.66666667,
    'D# Minor': -8.33333333, 'E Minor': -6.33333333, 'F Minor': -7.33333333,
    'F# Minor': -7, 'G Minor': -7.33333333, 'G# Minor': -8.33333333,
    'A Minor': -6.66666667, 'A# Minor': -8.33333333, 'B Minor': -7
    }
UNKNOWN_KEY_SCORE = 0

# --- 분위기 매핑을 위한 헬퍼 함수 ---
def calculate_distance(features, centroids):
    distance = 0
    for feature_name, feature_value in features.items():
        if feature_name in centroids:
            distance += (feature_value - centroids[feature_name]) ** 2
            
    return np.sqrt(distance)

def map_mood(features, centroids_map, feature_keys, what):
    min_distance = float('inf')
    mapped_mood = "알 수 없음"
    TANH_STEEPNESS = 0.3 # 탄젠트 함수 input 크기조절 
    MAX_ADJUSTMENT_PERCENTAGE = 0.2 # 길이에 몇% 적용시킬지. 0.2 = +-20%
    current_key_score = None
    if what == 'Valence':
        current_key_score = KEY_SCORES.get(features.get("Key", "Unknown"), UNKNOWN_KEY_SCORE)

    for mood, centroid_values in centroids_map.items():
        current_features_for_mood = {k: features.get(k) for k in feature_keys if k in features and features.get(k) is not None} # 특징명:값, 특징명:값
        centroid_values_for_mood = {k: centroid_values.get(k) for k in feature_keys if k in centroid_values}

        if len(current_features_for_mood) != len(feature_keys):
            print(f"경고: {mood} 매핑을 위한 일부 특징 값이 누락되었습니다.")
            continue

        if not all(k in current_features_for_mood and k in centroid_values_for_mood for k in feature_keys):
            continue

        current_distance = calculate_distance(current_features_for_mood, centroid_values_for_mood)
        adjusted_distance = current_distance

        if what == "Valence":
            if current_key_score is not None: 
                scaled_key_score = current_key_score * TANH_STEEPNESS
                tanh_output = math.tanh(scaled_key_score)
                percentage_adjustment = abs(tanh_output) * MAX_ADJUSTMENT_PERCENTAGE

                if tanh_output > 0:
                    adjusted_distance = current_distance * (1 + percentage_adjustment)
                elif tanh_output < 0:
                    adjusted_distance = current_distance * (1 - percentage_adjustment)
                
                adjusted_distance = max(0, adjusted_distance)

        if adjusted_distance < min_distance:
            min_distance = adjusted_distance
            mapped_mood = mood
            
    return mapped_mood

def map_valence_mood(audio_features):
    feature_keys = ["SpectralCentroid_Mean", "Spectral_Rolloff_Mean"]
    return map_mood(audio_features, VALENCE_CENTROIDS, feature_keys, "Valence")

def map_arousal_mood(audio_features):
    feature_keys = ["BPM", "RMS_Mean"]
    return map_mood(audio_features, AROUSAL_CENTROIDS, feature_keys, "Arousal")

def map_intensity_mood(audio_features):
    feature_keys = ["RMS_Mean", "SpectralFlatness_Mean"]
    return map_mood(audio_features, INTENSITY_CENTROIDS, feature_keys, "Intensity")

def map_complexity_mood(audio_features):
    feature_keys = ["RhythmComplexity_Score", "SpectralFlatness_Mean"]
    return map_mood(audio_features, COMPLEXITY_CENTROIDS, feature_keys, "Complexity")


def estimate_key(y, sr, file_name): #키 특징추출 함수
    if y is None or sr is None or len(y) == 0:
        return "unknown"
    
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=2048)
        chroma_mean = np.mean(chroma, axis=1)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        Keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        major_correlations = []
        minor_correlations = []

        if np.linalg.norm(chroma_mean) > 0:
            chroma_mean_normalized = chroma_mean / np.linalg.norm(chroma_mean)
        else:
            chroma_mean_normalized = chroma_mean

        major_profile_normalized = major_profile / np.linalg.norm(major_profile)
        minor_profile_normalized = minor_profile / np.linalg.norm(minor_profile)

        for i in range(12):
            shifted_chroma = np.roll(chroma_mean_normalized, -i)
            major_corr = np.dot(shifted_chroma, major_profile_normalized)
            minor_corr = np.dot(shifted_chroma, minor_profile_normalized)
            major_correlations.append(major_corr)
            minor_correlations.append(minor_corr)

        max_major_idx = np.argmax(major_correlations)
        max_minor_idx = np.argmax(minor_correlations)

        if major_correlations[max_major_idx] > minor_correlations[max_minor_idx]:
            estimated_key = f"{Keys[max_major_idx]} Major"
        else:
            estimated_key = f"{Keys[max_minor_idx]} Minor"
        
        return estimated_key

    except Exception as e:
        print(f"ERROR: {e}, '{file_name}' Key 추정 실패(함수 내부)")
        return "unknown"


def analyze_single_music_file(file_path):
    file_name = os.path.basename(file_path)
    print(f"'{file_name}' 분석 시작...")
    
    # --- 음악 파일 로딩 ---
    y = None
    sr = None
    
    try:
        y, sr = librosa.load(file_path, sr = sr)

        if len(y) == 0:
            print(f"WORNING: '{file_name}' 파일이 비어 있습니다. 분석을 건너뜁니다.")
            return None
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"ERROR: {e}, '{file_name} 로드 실패.")
        return None
    
    # --- 특징 추출 ---
    # BPM(tempo), Key, RMS_Mean, SpectralCentroid_Mean, SpectralFlatness_Mean, RhythmComplexity_Score, spectral_rolloff_mean
    tempo = 0.0
    avg_rms_energy = 0.0
    avg_spectral_centroid = 0.0
    estimated_key = "Unknown"
    flatness_mean = 0.0
    RhythmComplexity_Score = 0.0
    spectral_rolloff_mean = 0.0


    try: # tempo 추출
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(f"{tempo:.2f}")
    except Exception as e:
        print(f"ERROR: {e}, '{file_name}' tempo 추출 실패")

    try: # RMS_mean 추출
        rms_values = librosa.feature.rms(y=y)[0]
        avg_rms_energy = float(f"{np.mean(rms_values):.4f}")
    except Exception as e:
        print(f"ERROR: {e}, '{file_name}' RMS Energy 추출 실패")
    
    try: # Spectral Centroid 추출
        centroid_values = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_spectral_centroid = float(f"{np.mean(centroid_values):.4f}")
    except Exception as e:
        print(f"ERROR: {e}, '{file_name}' Spectral Centroid 추출 실패")
    
    try: # Key 추출
        estimated_key = estimate_key(y, sr, file_name)
    except Exception as e:
        print(f"ERROR: {e}, '{file_name}' Key 추정 실패(함수 외부)")

    try: # Spectral Flatness 추출
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        flatness_mean = np.mean(flatness)
    except Exception as e:
        print(f"ERROR: {e}, '{file_name}' Spectral Flatness 추출 실패")

    try: # Rhythm Complexity 추출
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        if len(onset_frames) < 2:
            RhythmComplexity_Score = 0.0
        else:
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            intervals = np.diff(onset_times)
            RhythmComplexity_Score = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0.0
    except Exception as e:
        print(f"ERROR: {e}, '{file_name}' Rhythm Complexity 추출 실패")
        RhythmComplexity_Score = 0.0
    
    try: # Spectral Rolloff 추출
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_rolloff_mean = np.mean(spectral_rolloff)
    except Exception as e:
        print(f"ERROR: {e}, '{file_name}' Spectral Rolloff 추출 실패")

    # --- 분위기 매핑 ---
    # 4가지 분위기 차원 : 구분에 사용할 특징
    # Valence(쾌감/가치)-밝음/어두움 : Key, SpectralCentroid_Mean, Spectral_Rolloff_Mean
    # Arousal(각성/활동성)-신남(활동적)/차분함(고요함) : BPM, RMS_Mean
    # Intensity(강도/밀도)-강렬함(힘있는)/평범함/부드러움(섬세함) : RMS_Mean, SpectralFlatness_Mean
    # Complexity(복잡성/밀도)-복잡한(다채로운)/규칙적인/단순한(반복적인) : RhythmComplexity_Score, SpectralFlatness_Mean

    # 특징 값들을 딕셔너리로 묶어 관리 (정규화는 필요 시 추가)
    audio_features = {
        "Key": estimated_key,
        "SpectralCentroid_Mean": avg_spectral_centroid,
        "Spectral_Rolloff_Mean": spectral_rolloff_mean,
        "BPM": tempo,
        "RMS_Mean": avg_rms_energy,
        "SpectralFlatness_Mean": flatness_mean,
        "RhythmComplexity_Score": RhythmComplexity_Score,
    }

    valence_mood = map_valence_mood(audio_features)
    arousal_mood = map_arousal_mood(audio_features)
    intensity_mood = map_intensity_mood(audio_features)
    complexity_mood = map_complexity_mood(audio_features)

    print(f"  - '{file_name}' 분석 완료: BPM={tempo:.2f}, Key={estimated_key}, "
          f"분위기= Valence: {valence_mood}, Arousal: {arousal_mood}, "
          f"Intensity: {intensity_mood}, Complexity: {complexity_mood}")

    # --- 7. 분석 결과 딕셔너리 생성 ---
    return {
        'FileName': file_name,
        'BPM': tempo,
        'Key': estimated_key,
        'RMS_Mean': avg_rms_energy,
        'SpectralCentroid_Mean': avg_spectral_centroid,
        'SpectralFlatness_Mean': flatness_mean,
        'RhythmComplexity_Score': RhythmComplexity_Score,
        'Spectral_Rolloff_Mean': spectral_rolloff_mean,
        'Valence_Mood': valence_mood,
        'Arousal_Mood': arousal_mood,
        'Intensity_Mood': intensity_mood,
        'Complexity_Mood': complexity_mood
    }


# =============================================================================
# 3. 메인 실행 로직
# =============================================================================
def main():
    print("--- 음악 파일 분석 프로그램 시작 ---")

    # 1. MUSIC_FOLDER 존재 여부 확인 및 생성
    if not os.path.exists(MUSIC_FOLDER):
        try:
            os.makedirs(MUSIC_FOLDER)
            print(f"폴더 '{MUSIC_FOLDER}'를 생성했습니다.")
            print(f"음악 파일을 '{MUSIC_FOLDER}' 폴더에 추가해주세요. (예: .mp3, .wav 등)")
            print("파일을 추가한 후 프로그램을 다시 실행해주세요.")
            sys.exit() # 폴더 생성 후 파일이 없으므로 프로그램 종료
        except Exception as e:
            print(f"ERROR: {e}, 폴더 '{MUSIC_FOLDER}' 생성 중 오류가 발생했습니다.")
            sys.exit(1) # 오류 발생 시 프로그램 종료

    all_analysis_results = []
    files_found = 0

    # 2. MUSIC_FOLDER 내의 모든 파일을 순회
    for root, _, files in os.walk(MUSIC_FOLDER):
        for file in files:
            # 3. 오디오 파일 확장자 확인
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                file_path = os.path.join(root, file)
                files_found += 1
                # 4. analyze_single_music_file 함수 호출하여 각 파일 분석
                result = analyze_single_music_file(file_path)
                # 5. 분석 성공 시 반환된 결과 딕셔너리를 all_analysis_results 리스트에 추가
                if result:
                    all_analysis_results.append(result)

    # 6. 분석할 파일이 없었거나 모든 파일 분석에 실패한 경우 처리
    if files_found == 0:
        print(f"경고: '{MUSIC_FOLDER}' 폴더에서 지원되는 음악 파일을 찾을 수 없습니다.")
        print("음악 파일을 폴더에 넣었는지 확인하거나, 지원되는 확장자를 확인해주세요.")
    elif not all_analysis_results: # 파일을 찾았지만 모두 분석 실패한 경우
        print("경고: 모든 음악 파일 분석에 실패했습니다. 로그를 확인해주세요.")

    # --- 8. 모든 분석 결과 취합 --- (all_analysis_results 리스트에 이미 취합됨)

    # --- 9. 결과 데이터프레임 생성 및 출력 ---
    if all_analysis_results:
        df_results = pd.DataFrame(all_analysis_results)
        print("\n--- 음악 파일 분석 결과 ---")
        print(df_results)
    else:
        print("\n분석된 음악 파일이 없어 결과를 출력할 수 없습니다.")


    # --- 10. 결과를 CSV 파일로 저장 ---
    if all_analysis_results:
        try:
            df_results.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
            print(f"\n분석 결과가 '{OUTPUT_CSV_FILE}' 파일로 성공적으로 저장되었습니다.")
        except Exception as e:
            print(f"ERROR: {e}, 분석 결과를 CSV 파일로 저장하는 데 실패했습니다.")
    else:
        print("\n분석된 음악 파일이 없어 CSV 파일로 저장하지 않습니다.")

    print("\n--- 음악 파일 분석 프로그램 종료 ---")

# =============================================================================
# 스크립트 실행 진입점
# =============================================================================
if __name__ == '__main__':
    main()