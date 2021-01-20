from descriptors import FeatureData, SimpleDescriptors, FourierDescriptor, SignatureDescriptors

if __name__ == "__main__":
    feature_obj = FeatureData('no_bg')
    print(feature_obj)
    feature_obj.show()
    feature_obj.bin('bin')
    train_paths, test_paths = feature_obj.split_test_train('training', bin=True)

    simple_desc = SimpleDescriptors(train_paths, test_paths, feature_obj)
    clf_1 = simple_desc.fit()
    results = simple_desc.test(clf_1)
    scores = simple_desc.score(results)
    print("\nPoprawność - pole powierzchni: ", scores[0], "%")
    print("Poprawność - obwód: ", scores[1], "%")
    print("Poprawność - kołowość: ", scores[2], "%")
    print("Poprawność - zwartość: ", scores[3], "%")
    print("Poprawność - obwód powłoki wypukłej: ", scores[4], "%")

    fft_sizes = [5, 10, 15, 20, 50, 100, 150, 200]
    fourier_desc = FourierDescriptor(train_paths, test_paths, feature_obj, fft_sizes)
    for fft_size in fft_sizes:
        clf = fourier_desc.fit(fft_size)
        results = fourier_desc.test(clf, fft_size)
        scores = sum(results) / len(results) * 100
        print(f'Poprawność - Fourier [{fft_size}x{fft_size}]: {scores}%')

    # signature_dest = SignatureDescriptors(train_paths, test_paths, feature_obj)
