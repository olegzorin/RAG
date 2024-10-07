from typing import List

from pydantic import BaseModel, TypeAdapter

DOCS_DIR = '../docs'

doc_names = [
    'CCR',
    'LHS',
    'MHR'
]


class TestConf(BaseModel):
    id: str
    params: dict[str, str]


class TestConfigs(BaseModel):
    params: dict[str, str]
    configs: list[TestConf]


test_configs = TypeAdapter(TestConfigs).validate_json(open('test_configs.json', 'r').read())


class Question(BaseModel):
    Key: str
    Value: str


questions: List[Question] = TypeAdapter(List[Question]).validate_json(open('questions.json', 'r').read())


class TestResult(BaseModel):
    doc_name: str
    conf_id: str
    chunk_size: int
    answers: list[str]

    def dump(self) -> str:
        return TypeAdapter(TestResult).dump_json(self).decode('utf-8')


def get_test_results(
        path: str
) -> list[TestResult]:
    with open(path, 'r') as f:
        return [TypeAdapter(TestResult).validate_json(line) for line in f.readlines()]
