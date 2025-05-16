import pandas as pd
import numpy as np
import json

# Hàm tải mô hình từ tệp JSON
def load_model(filename='hmm_model.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        model = json.load(f)
    A = np.array(model['transition_matrix_A'])
    B = np.array(model['emission_matrix_B'])
    pi = np.array(model['initial_distribution_pi'])
    return A, B, pi

# Hàm phân loại số ca sinh
def classify_birth(birth, bins):
    if birth <= bins[0]:
        return 0  # Thấp
    elif birth <= bins[1]:
        return 1  # Trung bình
    else:
        return 2  # Cao

# Thuật toán Viterbi để suy ra trạng thái ẩn
def viterbi(obs, A, B, pi):
    T = len(obs)
    N = len(pi)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    states = np.zeros(T, dtype=int)
    
    # Khởi tạo
    for i in range(N):
        delta[0, i] = pi[i] * B[i, obs[0]]
    
    # Đệ quy
    for t in range(1, T):
        for j in range(N):
            max_val = 0
            max_idx = 0
            for i in range(N):
                val = delta[t-1, i] * A[i, j]
                if val > max_val:
                    max_val = val
                    max_idx = i
            delta[t, j] = max_val * B[j, obs[t]]
            psi[t, j] = max_idx
    
    # Kết thúc
    states[T-1] = np.argmax(delta[T-1, :])
    
    # Backtracking
    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]
    
    return states

# Hàm chính để áp dụng mô hình HMM
def apply_hmm_model(data_file='DailyTotalFemaleBirths.csv', model_file='hmm_model.json', output_file='hidden_states_output.csv'):
    # Tải mô hình
    A, B, pi = load_model(model_file)
    
    # Đọc dữ liệu
    df = pd.read_csv(data_file)
    births = df['Births'].values
    
    # Phân loại số ca sinh
    bins = [np.percentile(births, 33), np.percentile(births, 66)]
    observations = [classify_birth(b, bins) for b in births]
    
    # Suy ra trạng thái ẩn
    hidden_states = viterbi(observations, A, B, pi)
    
    # Tạo DataFrame kết quả
    result_df = pd.DataFrame({
        'Date': df['Date'],
        'Births': df['Births'],
        'Hidden_State': hidden_states,
        'State_Label': [ {0: 'Thấp', 1: 'Trung bình', 2: 'Cao'}[s] for s in hidden_states ]
    })
    
    # Lưu kết quả vào CSV
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # In thông tin
    print("Đã tải mô hình từ", model_file)
    print("\nMa trận chuyển trạng thái (A):")
    print(A)
    print("\nMa trận phát sinh (B):")
    print(B)
    print("\nPhân phối trạng thái ban đầu (pi):")
    print(pi)
    print("\n10 dòng đầu tiên của kết quả:")
    print(result_df.head(10))
    print(f"\nKết quả đã được lưu vào '{output_file}'")

# Chạy chương trình
if __name__ == "__main__":
    apply_hmm_model()