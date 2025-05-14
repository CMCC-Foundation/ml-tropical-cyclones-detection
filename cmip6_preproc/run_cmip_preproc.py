from cmip_preproc import CMIP6Preprocessor

if __name__ == "__main__":
    processor = CMIP6Preprocessor("config_cmip_preproc.toml")
    processor.run()