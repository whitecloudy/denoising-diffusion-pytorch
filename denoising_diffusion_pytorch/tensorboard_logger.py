from torch.utils.tensorboard import SummaryWriter, writer


def read_python_file_cleaned(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    # 1. 멀티라인 주석 제거 (''' ''' 또는 """ """)
    code = re.sub(r"'''[\s\S]*?'''", '', code)
    code = re.sub(r'"""[\s\S]*?"""', '', code)

    # 2. 한 줄 주석 제거 (// 또는 #)
    code = re.sub(r'#.*', '', code)

    # 3. 여러 줄 공백을 하나의 줄로 압축
    code = re.sub(r'\n\s*\n+', '\n\n', code)

    # 4. 양 끝 공백 제거
    cleaned_code = code.strip()

    # 5. Markdown 코드 블록 포맷으로 감싸기
    markdown_code_block = f"\n```python\n{cleaned_code}\n```"
    return markdown_code_block


class Tensorboard_logger(SummaryWriter):
    def __init__(self,
                log_dir : str,
                *,
                enable=True):
        if enbale:
            super().__init__(log_dir=log_dir)
            import git
            repo = git.Repo(search_parent_directories=True)
            super().add_text('git_info', 
                                        f'commit: {repo.head.commit.hexsha}\nbranch: {repo.active_branch.name}\ndirty: {repo.is_dirty()}')
            import sys
            super().add_text('python_info', 
                                        f'python version: {sys.version}\nfile: {read_python_file_cleaned(sys.argv[0])}')
        else:
            super().__init__(log_dir="/dev/null")
        
    def __del__():
        super().close()