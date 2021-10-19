import pathlib


class DataOutput:
    __directory: str
    __file_prefix: str
    __file_suffix: str

    def __init__(
        self, directory: str, file_prefix: str = None, file_suffix: str = None
    ):
        self.__file_prefix = file_prefix
        self.__file_suffix = file_suffix
        output_dir = pathlib.Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.__directory = directory

    def __createFileName(self, name, extension=None):
        s = self.__directory + "/"
        if self.__file_prefix is not None and len(self.__file_prefix) > 0:
            s = self.__file_prefix + "_" + s
        s += name
        if self.__file_suffix is not None and len(self.__file_suffix) > 0:
            s += "_" + self.__file_suffix
        if extension is not None and len(extension) > 0:
            s += "." + extension
        return s
