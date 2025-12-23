# Vars to Explore: 
# - clouds
# - fog (maybe)?? but skews right quite a bit
# - bright
# - beautiful? more uniform distributions here
# - soothing (not super skewed though and not quite sure what this is a measure of!)
# - dirty??
# - sunrisesunset
# - dawndusk


from os.path import join
import pandas as pd
import matplotlib.pyplot as plt


BASE_PATH = 'data'
label = 'clouds'


def visualize_dataset(db="skyfinder"):
    file_path = join(BASE_PATH, "skyfinder.csv")
    data = pd.read_csv(file_path)
    data = data.dropna(subset=[label])
    data = data[data['CamId'].isin([5021, 3395, 10870, 4584])].copy()
    data[label] = (data[label]*100).astype(int)
    _, ax = plt.subplots(figsize=(6, 3), sharex='all', sharey='all')
    ax.hist(data[label], bins=range(int(max(data[label])) + 2))
    # ax.set_xlim([0, 102])
    plt.title(f"{db.upper()} (total: {data.shape[0]})")
    plt.tight_layout()
    plt.show()


def make_balanced_testset(db="skyfinder", max_size=20, seed=700, verbose=True, vis=True, save=False):
    file_path = join(BASE_PATH, f"{db}.csv")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=[label])
    df = df[df['CamId'].isin([5021, 3395, 10870, 4584])].copy()
    # only use cam id 5021
    df[label] = (df[label]*100).astype(int)
    val_set, test_set = [], []
    import random
    random.seed(seed)
    for value in range(101):
        curr_df = df[df[label] == value]
        curr_data = curr_df['Filename'].values
        random.shuffle(curr_data)
        curr_size = min(len(curr_data) // 3, max_size)
        val_set += list(curr_data[:curr_size])
        test_set += list(curr_data[curr_size:curr_size * 2])

    if verbose:
        print(f"Val: {len(val_set)}\nTest: {len(test_set)}")
    assert len(set(val_set).intersection(set(test_set))) == 0
    combined_set = dict(zip(val_set, ['val' for _ in range(len(val_set))]))
    combined_set.update(dict(zip(test_set, ['test' for _ in range(len(test_set))])))
    df['split'] = df['Filename'].map(combined_set)
    df['split'].fillna('tdirty', inplace=True)
    if verbose:
        print(df)
    if save:
        df.to_csv(str(join(BASE_PATH, f"{db}.csv")), index=False)
    if vis:
        _, ax = plt.subplots(3, figsize=(6, 9), sharex='all')
        df_tdirty = df[df['split'] == 'tdirty']
        ax[0].hist(df_tdirty[label], range(max(df[label])))
        ax[0].set_title(f"[{db.upper()}] tdirty: {df_tdirty.shape[0]}")
        ax[1].hist(df[df['split'] == 'val'][label], range(max(df[label])))
        ax[1].set_title(f"[{db.upper()}] val: {df[df['split'] == 'val'].shape[0]}")
        ax[2].hist(df[df['split'] == 'test'][label], range(max(df[label])))
        ax[2].set_title(f"[{db.upper()}] test: {df[df['split'] == 'test'].shape[0]}")
        ax[0].set_xlim([0, 101])
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    make_balanced_testset()
    visualize_dataset()