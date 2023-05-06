import configparser


class ReversiConfigSection(object):
    def __init__(self, section: configparser.SectionProxy) -> None:
        self.__section = section

    @staticmethod
    def getRealType(s: str):
        if s.lower() == "true":
            return True
        elif s.lower() == "false":
            return False
        if s.isdigit():
            return int(s)
        else:
            try:
                return float(s)
            except ValueError:
                return s

    @staticmethod
    def isSupportedTypes(__value) -> bool:
        return isinstance(__value, (int, float, str, bool))

    def __getattr__(self, key: str) -> str:
        return self.getRealType(self.__section[key])

    def __setattr__(self, __name: str, __value) -> None:
        if not self.isSupportedTypes(__value):
            object.__setattr__(self, __name, __value)
            return

        self.__section[__name] = str(__value)

# classes below are only used in fake type annotations


class PathSection(ReversiConfigSection):
    MCTSFolder: str
    ModelBaseName: str
    MCTSModelPath: str
    PureMCTSProgramPath: str
    MCTSProgramPath: str
    DataFolder: str
    L2FileName: str
    LossFileName: str
    GameRecordFile: str


class MCTSSection(ReversiConfigSection):
    LogLevel: str
    Port: int
    DivideLevel: int
    StartArguments: str
    BaseArguments: str
    Arguments: str
    PerfTestArguments: str
    PureMCTSPort: int
    PureMCTSStartArguments: str
    PureMCTSArguments: str


class BenchmarkSection(ReversiConfigSection):
    MaxRound: int
    Test1Port: int
    Test2Port: int
    Test1OnnxBaseName: str
    Test2OnnxBaseName: str
    TestOnnxFolder: str
    BenchmarkBaseStartArg: str
    BenchmarkTestStartArg1: str
    BenchmarkTestStartArg2: str
    Temperature: float
    DivideLevel: int
    VisitCount: int
    BaseArguments: str


class TorchModelSection(ReversiConfigSection):
    SaveDir: str
    BaseSaveName: str
    SaveInterval: int
    RecordBatchSize: int
    BatchNum: int


class ResNetParamSection(ReversiConfigSection):
    LR: float
    L2: float
    TotalResblockNum: int
    InitialChannelNum: int


class GeneralSection(ReversiConfigSection):
    IsContinueTraining: bool


class CheckPointSection(ReversiConfigSection):
    Epoch: int
    LR: float


class PlotSection(ReversiConfigSection):
    PlotInterval: int


class SelectorSection(ReversiConfigSection):
    DataFile: str
    Port: int


class ReversiConfig(configparser.ConfigParser):
    """
    A wrapper of `configparser.ConfigParser` to make it more convenient to use.
    You can always use `.` instead of `[]` to access the config.
    If you need to add more section in the config file, you should add a new class 
    inheriting `ReversiConfigSection` and add its item annotation, and then 
    add it to the type annotation of `ReversiConfig`.
    """

    def __init__(self, filename: str) -> None:
        self.__reversi_cfg_filename = filename
        super().__init__(interpolation=configparser.ExtendedInterpolation())
        self.read(filename)

    def __getattr__(self, section: str) -> ReversiConfigSection:
        if not self.has_section(section):
            self.add_section(section)
        return ReversiConfigSection(self[section])

    def save(self):
        with open(self.__reversi_cfg_filename, 'w') as f:
            self.write(f)

    def optionxform(self, optionstr):
        return optionstr


class ReversiGeneralConfig(ReversiConfig):
    # these type annotations are fake; real type is always ReversiConfigSection
    Paths: PathSection
    MCTS: MCTSSection
    Benchmark: BenchmarkSection
    TorchModel: TorchModelSection
    ResNetParam: ResNetParamSection
    General: GeneralSection
    Plot: PlotSection
    Selector: SelectorSection


class ReversiCheckPointConfig(ReversiConfig):
    # these type annotations are fake; real type is always ReversiConfigSection
    CheckPoint: CheckPointSection
