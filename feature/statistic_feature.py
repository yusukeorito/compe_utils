class GroupingEngine: 

    def __init__(self, group_key, group_values, agg_methods):
        self.group_key = group_key
        self.group_values = group_values

        ex_trans_methods = ["val-mean", "z-score"]
        self.ex_trans_methods = [m for m in agg_methods if m in ex_trans_methods]
        self.agg_methods = [m for m in agg_methods if m not in self.ex_trans_methods]
        self.df = None

    def fit(self, input_df, y=None):
        new_df = []
        for agg_method in self.agg_methods:

            for col in self.group_values:
                if callable(agg_method):
                    agg_method_name = agg_method.__name__
                else:
                    agg_method_name = agg_method

                new_col = f"agg_{agg_method_name}_{col}_grpby_{self.group_key}"
                df_agg = (input_df[[col] + [self.group_key]].groupby(self.group_key)[[col]].agg(agg_method))
                df_agg.columns = [new_col]
                new_df.append(df_agg)
        self.df = pd.concat(new_df, axis=1).reset_index()

    def transform(self, input_df):
        output_df = pd.merge(input_df[[self.group_key]], self.df, on=self.group_key, how="left")
        if len(self.ex_trans_methods) != 0:
            output_df = self.ex_transform(input_df, output_df)
        output_df.drop(self.group_key, axis=1, inplace=True)
        return output_df

    def ex_transform(self, df1, df2):
        """
        df1: input_df
        df2: output_df
        return: output_df (added ex transformed features)
        """

        if "val-mean" in self.ex_trans_methods:
            df2[self._get_col("val-mean")] = df1[self.group_values].values - df2[self._get_col("mean")].values
        if "z-score" in self.ex_trans_methods:
            df2[self._get_col("z-score")] = (df1[self.group_values].values - df2[self._get_col("mean")].values) \
                                            / (df2[self._get_col("std")].values + 1e-3)
        return df2

    def _get_col(self, method):
        return np.sort([f"agg_{method}_{group_val}_grpby_{self.group_key}" for group_val in self.group_values])

    def fit_transform(self, input_df, y=None):
        self.fit(input_df, y=y)
        return self.transform(input_df)  