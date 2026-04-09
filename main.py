import torchPIV

if __name__ == "__main__":
    try:
        import pyi_splash
        pyi_splash.close()
    except ImportError:
        pass
    torchPIV.runGUI()