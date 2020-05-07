package net.mnementh64.neural.utils;

import net.mnementh64.neural.model.DataRow;

import java.util.List;
import java.util.stream.Collectors;

public class DataRowUtils {

    public static List<DataRow> extractTrainingDataRow(List<DataRow> data, double percentTraining) throws Exception {
        int limit = (int) Math.floor(data.size() * percentTraining);
        return data.stream()
                .limit(limit)
                .collect(Collectors.toList());
    }

    public static List<DataRow> extractGeneralizeDataRow(List<DataRow> data, double percentTraining) throws Exception {
        int limitTraining = (int) Math.floor(data.size() * percentTraining);
        return data.stream()
                .skip(limitTraining)
                .collect(Collectors.toList());
    }

}
