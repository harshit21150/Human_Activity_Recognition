# Function to extract features for a single frame
def convo_features(frame):
    img = image.array_to_img(frame, scale=False)
    img = img.resize((220, 220))  # Adjust the target size as needed
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    return features.flatten()

# Function to classify a video
def classify_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (220, 220))

        frame_features = convo_features(frame)
        flattened_features = frame_features.flatten()

        if flattened_features.shape[0] != loaded_pca_model.n_features_in_:
            print(f"Number of features ({flattened_features.shape[0]}) doesn't match expected features ({loaded_pca_model.n_features_in_}). Check your feature extraction and PCA model.")

        pca_features = loaded_pca_model.transform(flattened_features.reshape(1, -1))
        predicted_class = loaded_svm_model.predict(pca_features)

        frames.append(predicted_class[0])

    cap.release()
    return frames

# Example usage:
video_path = '/kaggle/input/laughing/video (2160p).mp4'
predicted_classes = classify_video(video_path)

print("Predicted classes for each frame:", predicted_classes)

final_dict={}
for i in set(predicted_classes):
    counter=0
    for k in predicted_classes:
        if i==k:
            counter+=1
    final_dict[i]=counter
    
res=max(zip(final_dict.keys(), final_dict.values()))[1]
    
