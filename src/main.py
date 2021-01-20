from descriptors import FeatureData, SimpleDescriptors, FourierDescriptor, SignatureDescriptors, UnlFourierDescriptor

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
    print("-------------------------------------------------------------")
    print("\nPoprawność - pole powierzchni: ", scores[0], "%")
    print("Poprawność - obwód: ", scores[1], "%")
    print("Poprawność - kołowość: ", scores[2], "%")
    print("Poprawność - zwartość: ", scores[3], "%")
    print("Poprawność - obwód powłoki wypukłej: ", scores[4], "%")
    print("-------------------------------------------------------------")

    fft_sizes = [5, 10, 15, 20, 50, 100, 150, 200]
    fourier_desc = FourierDescriptor(train_paths, test_paths, feature_obj, fft_sizes)
    print("-------------------------------------------------------------")
    for fft_size in fft_sizes:
        clf = fourier_desc.fit(fft_size)
        results = fourier_desc.test(clf, fft_size)
        scores = sum(results) / len(results) * 100
        print(f'Poprawność - Fourier [{fft_size}x{fft_size}]: {scores}%')
    print("-------------------------------------------------------------")
    signature_dest = SignatureDescriptors(train_paths, test_paths, feature_obj)
    clf = signature_dest.fit()
    results = signature_dest.test(clf)
    scores = signature_dest.score(results)
    print("-------------------------------------------------------------")
    print("Poprawność - Sygnatura: ", scores, "%")
    print("-------------------------------------------------------------")

    sizes = [5, 10, 15, 20, 30, 50, 100]
    unl_fourier_desc = UnlFourierDescriptor(train_paths, test_paths, feature_obj, sizes)
    print("-------------------------------------------------------------")
    for s in sizes:
        clf = unl_fourier_desc.fit(s)
        results = unl_fourier_desc.test(clf, s)
        scores = unl_fourier_desc.score(results)
        print(f'Poprawność - UNL-Fourier [{s}x{s}]: {scores}%')
    print("-------------------------------------------------------------")
