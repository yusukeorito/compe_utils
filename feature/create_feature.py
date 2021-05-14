import pandas as pd

def decorate(s: str, decoration=None):
    if decoration is None:
        decoration = '★' * 20

    return ' '.join([decoration, str(s), decoration])

def run_blocks(input_df, blocks, y=None, test=False):
    output_df = pd.DataFrame()

    print(decorate('start run blocks...'))

    with Timer(prefix='run test={}'.format(test)):
        for block in blocks:
            with Timer(prefix='\t-{}'.format(str(block))):
                if not test:
                    out_i = block.fit(input_df, y=y)
                else:
                    out_i = block.transform(input_df)

            assert len(input_df) == len(out_i), block
            name = block.__class__.__name__
            output_df = pd.concat([output_df, out_i.add_suffix(f'@{name}')], axis=1)
    return output_df



