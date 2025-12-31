try:
    import medmnist
    from medmnist import INFO, load
    print("medmnist import OK. Datasets:", len(INFO))
except Exception as e:
    print("medmnist import failed:", repr(e))

