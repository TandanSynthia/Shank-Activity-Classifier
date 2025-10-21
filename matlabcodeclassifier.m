% SHANK CLASSIFIER PIPELINE (LOSO, 10 Hz, 4-class: Standing, Stepping, Transport, and Non-wear)
% Orientation-Invariant Activity Classifier for Shank-Mounted Accelerometers
%
% Purpose:
%   This script performs a Leave-One-Subject-Out (LOSO) Random Forest classification
%   of shank-mounted accelerometer data. The aim is to identify key activity
%   states—standing, stepping, in transport, and non-wear—using orientation-
%   invariant magnitude features.
% Classes:
%   1 = Upright/Static
%   2 = Stepping
%   4 = Non-wear
%   5 = Transport
%
% Key Features:
%   10-second short-term features (from acceleration magnitude)
%   60-minute contextualRolling mean of each short-term feature
%   Orientation invariant (based on magnitude only)
%   LOSO validation across all subjects
%   Outputs accuracy, F1-score, and confusion matrix
%
% Output:
%   LOSO performance metrics for each participant
%   Confusion matrix visualization
%    Trained Random Forest model (.mat file) with scaling parameters
% =========================================================================

clear; clc; close all; rng(13);  % Clear workspace and set random seed

%% USER SETTINGS
% Define analysis parameters for the pipeline
numberOfParticipants = 14;           % Number of participants to include
windowDurationSeconds = 10;         % Short-term window size (seconds)
contextWindowMinutes = 60;          % Long-term contextual window (minutes)
minimumLabelCoverage = 0.6;         % Minimum label dominance per window
limitToFirstDays = [];               % Restrict analysis to the first N days (set [] for all)
targetSamplingRateHz = 10;          % Target sampling frequency (Hz)

%% FOLDER SELECTION
% Prompt user to select folders for accelerometer and event CSVs
accelerometerFolder = uigetdir(pwd, 'Select folder containing accelerometer CSV files');
if accelerometerFolder == 0, error('No accelerometer folder selected.'); end

eventFolder = uigetdir(pwd, 'Select folder containing event CSV files');
if eventFolder == 0, error('No event folder selected.'); end

% Load file lists from both folders
accelerometerFiles = dir(fullfile(accelerometerFolder, '*Accelerometer*.csv'));
eventFiles = dir(fullfile(eventFolder, '*Event*.csv'));
if isempty(accelerometerFiles) || isempty(eventFiles)
    error('No CSV files found in selected folders.');
end

% Automatically map accelerometer and event files by order
numberOfParticipants = min(numel(accelerometerFiles), numel(eventFiles));
participantFilePairs = cell(numberOfParticipants, 2);
for i = 1:numberOfParticipants
    participantFilePairs{i,1} = fullfile(accelerometerFiles(i).folder, accelerometerFiles(i).name);
    participantFilePairs{i,2} = fullfile(eventFiles(i).folder, eventFiles(i).name);
end

fprintf('Found %d participant file pairs.\n', numberOfParticipants);

% Display file mapping summary for user verification
participantMappingTable = table((1:numberOfParticipants)', ...
    string({accelerometerFiles.name})', string({eventFiles.name})', ...
    'VariableNames', {'Subject', 'Accelerometer_File', 'Event_File'});
disp(participantMappingTable);

%% PARTICIPANT PREPARATION
% Process each participant's accelerometer and event data to extract labeled features
fprintf('\nPreparing participant data...\n');
ParticipantResults(numberOfParticipants) = struct('FeatureMatrix', [], 'LabelVector', [], 'TimeVector', []);

for subjectIndex = 1:numberOfParticipants
    ParticipantResults(subjectIndex) = processParticipantFiles( ...
        participantFilePairs{subjectIndex,1}, participantFilePairs{subjectIndex,2}, ...
        windowDurationSeconds, contextWindowMinutes, minimumLabelCoverage, ...
        limitToFirstDays, targetSamplingRateHz, subjectIndex);
end

% Remove invalid or empty participants (e.g., missing or corrupted data)
validParticipants = arrayfun(@(p) ~isempty(p.FeatureMatrix) && all(isfinite(p.FeatureMatrix(:))), ParticipantResults);
ParticipantResults = ParticipantResults(validParticipants);
numberOfParticipants = sum(validParticipants);
fprintf('Valid participants remaining: %d\n', numberOfParticipants);

%% LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION (LOSO)
% Each subject is left out once as a test set, while the others form the training set
fprintf('\nPerforming Leave-One-Subject-Out Cross-validation...\n');
accuracyScores = nan(numberOfParticipants, 1);
macroF1Scores = nan(numberOfParticipants, 1);
CrossValidationResults(numberOfParticipants) = struct('TrueLabels', [], 'PredictedLabels', [], ...
    'Accuracy', [], 'F1', [], 'SampleCount', []);

for testSubjectIndex = 1:numberOfParticipants

    % Split into training and testing data
    TrainingFeatures = [];
    TrainingLabels = [];
    for subjectIndex = 1:numberOfParticipants
        if subjectIndex == testSubjectIndex, continue; end
        TrainingFeatures = [TrainingFeatures; ParticipantResults(subjectIndex).FeatureMatrix];
        TrainingLabels = [TrainingLabels; ParticipantResults(subjectIndex).LabelVector];
    end
    TestingFeatures = ParticipantResults(testSubjectIndex).FeatureMatrix;
    TestingLabels = ParticipantResults(testSubjectIndex).LabelVector;
    if isempty(TrainingFeatures) || isempty(TestingFeatures), continue; end

    % Normalize features (zero mean, unit variance)
    meanTraining = mean(TrainingFeatures, 1);
    stdTraining = std(TrainingFeatures, 0, 1);
    stdTraining(stdTraining == 0) = 1;
    NormalizedTrainingFeatures = (TrainingFeatures - meanTraining) ./ stdTraining;
    NormalizedTestingFeatures = (TestingFeatures - meanTraining) ./ stdTraining;

    % Remove invalid samples (NaN or Inf)
    invalidTrain = any(~isfinite(NormalizedTrainingFeatures), 2);
    invalidTest  = any(~isfinite(NormalizedTestingFeatures), 2);
    NormalizedTrainingFeatures(invalidTrain,:) = [];
    TrainingLabels(invalidTrain,:) = [];
    NormalizedTestingFeatures(invalidTest,:) = [];
    TestingLabels(invalidTest,:) = [];
    if isempty(NormalizedTrainingFeatures) || isempty(NormalizedTestingFeatures), continue; end

    % Train Random Forest classifier
    RandomForestModel = TreeBagger(200, NormalizedTrainingFeatures, categorical(TrainingLabels), ...
        'Method', 'classification', 'MinLeafSize', 3, 'NumPredictorsToSample', 'all');

    % Predict on test subject
    PredictedLabels = str2double(predict(RandomForestModel, NormalizedTestingFeatures));

    % Calculate accuracy and F1 metrics
    accuracyScores(testSubjectIndex) = mean(PredictedLabels == TestingLabels) * 100;
    classList = [1 2 4 5];
    f1PerClass = nan(1, numel(classList));
    for classIndex = 1:numel(classList)
        classID = classList(classIndex);
        truePositive = sum(PredictedLabels == classID & TestingLabels == classID);
        falsePositive = sum(PredictedLabels == classID & TestingLabels ~= classID);
        falseNegative = sum(PredictedLabels ~= classID & TestingLabels == classID);
        precision = truePositive / (truePositive + falsePositive + eps);
        recall = truePositive / (truePositive + falseNegative + eps);
        f1PerClass(classIndex) = 2 * precision * recall / (precision + recall + eps);
    end
    macroF1Scores(testSubjectIndex) = mean(f1PerClass, 'omitnan');

    % Store results
    CrossValidationResults(testSubjectIndex).TrueLabels = TestingLabels;
    CrossValidationResults(testSubjectIndex).PredictedLabels = PredictedLabels;
    CrossValidationResults(testSubjectIndex).Accuracy = accuracyScores(testSubjectIndex);
    CrossValidationResults(testSubjectIndex).F1 = macroF1Scores(testSubjectIndex);
    CrossValidationResults(testSubjectIndex).SampleCount = numel(TestingLabels);
end

% Display per-subject results and compute mean LOSO metrics
AllTrueLabels = [];
AllPredictedLabels = [];
for subjectIndex = 1:numberOfParticipants
    if isempty(CrossValidationResults(subjectIndex).TrueLabels), continue; end
    fprintf('Subject %d — Accuracy %.2f%% | F1 %.3f | n = %d\n', ...
        subjectIndex, CrossValidationResults(subjectIndex).Accuracy, ...
        CrossValidationResults(subjectIndex).F1, CrossValidationResults(subjectIndex).SampleCount);
    AllTrueLabels = [AllTrueLabels; CrossValidationResults(subjectIndex).TrueLabels];
    AllPredictedLabels = [AllPredictedLabels; CrossValidationResults(subjectIndex).PredictedLabels];
end
fprintf('\nMean LOSO Accuracy: %.2f%% | Mean Macro-F1: %.3f\n', ...
    mean(accuracyScores, 'omitnan'), mean(macroF1Scores, 'omitnan'));

%% CONFUSION MATRIX VISUALIZATION
% Generate overall confusion matrix combining all LOSO predictions
fprintf('\nGenerating Confusion Matrix...\n');
classList = [1 2 4 5];
ConfusionMatrix = confusionmat(AllTrueLabels, AllPredictedLabels, 'Order', classList);
overallAccuracy = 100 * sum(diag(ConfusionMatrix)) / sum(ConfusionMatrix(:));

figure('Name', 'Overall Confusion Matrix', 'Position', [100 100 900 600]);
confusionchart(AllTrueLabels, AllPredictedLabels, ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized', ...
    'Title', sprintf('Overall Confusion Matrix (%.1f%% Accuracy)', overallAccuracy), ...
    'XLabel', 'Predicted Class', 'YLabel', 'True Class');

%% SAVE FINAL MODEL
% Retrain final Random Forest model using all available data
fprintf('\nTraining Final Model on All Data...\n');
AllFeatures = [];
AllLabels = [];
for subjectIndex = 1:numberOfParticipants
    AllFeatures = [AllFeatures; ParticipantResults(subjectIndex).FeatureMatrix];
    AllLabels = [AllLabels; ParticipantResults(subjectIndex).LabelVector];
end

% Normalize complete dataset
meanAllFeatures = mean(AllFeatures, 1);
stdAllFeatures = std(AllFeatures, 0, 1);
stdAllFeatures(stdAllFeatures == 0) = 1;
NormalizedAllFeatures = (AllFeatures - meanAllFeatures) ./ stdAllFeatures;

% Train final Random Forest on all data
FinalModel = TreeBagger(300, NormalizedAllFeatures, categorical(AllLabels), ...
    'Method', 'classification', 'MinLeafSize', 3, 'NumPredictorsToSample', 'all');

% Save model and scaling parameters
[fileName, filePath] = uiputfile('*.mat', 'Save trained classifier as', ...
    sprintf('Shank_RF_LOSO_10Hz_%s.mat', datestr(now, 'yyyymmdd_HHMM')));
if ~isequal(fileName, 0)
    FeatureScaler.mean = meanAllFeatures;
    FeatureScaler.std = stdAllFeatures;
    save(fullfile(filePath, fileName), 'FinalModel', 'FeatureScaler', '-v7.3');
    fprintf('Saved final model: %s\n', fullfile(filePath, fileName));
end

%%subfunction
function ParticipantData = processParticipantFiles(accelFile, eventFile, windowDurationSeconds, contextWindowMinutes, minimumLabelCoverage, limitToFirstDays, targetSamplingRateHz, subjectIndex)
% This function processes a single participant's accelerometer and event data:
% 1. Reads accelerometer and event CSV files
% 2. Aligns timestamps between both sources
% 3. Segments accelerometer data into fixed-size time windows
% 4. Assigns a dominant event label to each window
% 5. Computes 10 magnitude-based features per window
% 6. Adds contextual rolling means over 60 minutes

ParticipantData = struct('FeatureMatrix', [], 'LabelVector', [], 'TimeVector', []);
try
    fprintf('Processing Participant %d\n', subjectIndex);

    % --- Read accelerometer file ---
    AccelerometerTable = readtable(accelFile, 'VariableNamingRule', 'preserve');
    variableNames = lower(string(AccelerometerTable.Properties.VariableNames));
    timeColumn = find(ismember(variableNames, ["time","timestamp","datetime","time(approx)"]), 1);
    xColumn = find(ismember(variableNames, ["x","accel_x","ax"]), 1);
    yColumn = find(ismember(variableNames, ["y","accel_y","ay"]), 1);
    zColumn = find(ismember(variableNames, ["z","accel_z","az"]), 1);
    assert(~isempty([timeColumn xColumn yColumn zColumn]), 'Missing accelerometer columns');

    % Parse timestamps and clean data
    timeVector = parseTimeFlexible(AccelerometerTable.(timeColumn));
    timeVector.TimeZone = '';
    TimeTable = timetable(timeVector, AccelerometerTable.(xColumn), ...
        AccelerometerTable.(yColumn), AccelerometerTable.(zColumn), ...
        'VariableNames', {'X', 'Y', 'Z'});
    TimeTable = sortrows(TimeTable);
    TimeTable = fillmissing(TimeTable, 'linear', 'MaxGap', seconds(3));
    TimeTable = rmmissing(TimeTable);

    % Restrict data to first N days if specified
    if ~isempty(limitToFirstDays)
        startTime = TimeTable.Properties.RowTimes(1);
        TimeTable = TimeTable(TimeTable.Properties.RowTimes <= startTime + days(limitToFirstDays), :);
    end
    if isempty(TimeTable), return; end

    % --- Read event annotation file ---
    EventTable = readtable(eventFile, 'VariableNamingRule', 'preserve');
    eventVariableNames = string(EventTable.Properties.VariableNames);
    timeCol = find(contains(lower(eventVariableNames), "time"), 1);
    durationCol = find(contains(lower(eventVariableNames), "duration"), 1);
    eventCol = find(contains(lower(eventVariableNames), "event"), 1);

    eventStartTimes = parseTimeFlexible(EventTable.(timeCol));
    eventDurations = seconds(EventTable.(durationCol));
    eventEndTimes = eventStartTimes + eventDurations;
    eventLabels = double(str2double(string(EventTable.(eventCol))));

    % Clip accelerometer time range to event overlap
    accelStart = TimeTable.Properties.RowTimes(1);
    accelEnd = TimeTable.Properties.RowTimes(end);
    analysisStart = max(accelStart, min(eventStartTimes));
    analysisEnd = min(accelEnd, max(eventEndTimes));
    TimeTable = TimeTable(TimeTable.Properties.RowTimes >= analysisStart & ...
        TimeTable.Properties.RowTimes <= analysisEnd, :);
    if isempty(TimeTable), return; end

    % --- Align labels to accelerometer timestamps ---
    sampleTimes = TimeTable.Properties.RowTimes;
    alignedLabels = zeros(height(TimeTable), 1);
    for k = 1:numel(eventLabels)
        indices = (sampleTimes >= eventStartTimes(k)) & (sampleTimes < eventEndTimes(k));
        if any(indices)
            labelValue = floor(eventLabels(k));
            if ismember(labelValue, [1 2 4 5])
                alignedLabels(indices) = labelValue;
            end
        end
    end

    % --- Segment accelerometer data into windows and extract features ---
    totalSamples = height(TimeTable);
    samplesPerWindow = windowDurationSeconds * targetSamplingRateHz;
    numberOfWindows = floor(totalSamples / samplesPerWindow);
    WindowedFeatures = [];
    WindowedLabels = [];
    WindowedTimes = [];

    for windowIndex = 1:numberOfWindows
        startIdx = (windowIndex - 1) * samplesPerWindow + 1;
        endIdx = windowIndex * samplesPerWindow;
        segmentX = TimeTable.X(startIdx:endIdx);
        segmentY = TimeTable.Y(startIdx:endIdx);
        segmentZ = TimeTable.Z(startIdx:endIdx);
        segmentLabels = alignedLabels(startIdx:endIdx);

        % Determine dominant label in window
        [uniqueLabels,~,indices] = unique(segmentLabels,'stable');
        counts = accumarray(indices,1);
        [maxCount, maxIndex] = max(counts);
        dominantLabel = uniqueLabels(maxIndex);

        % Skip windows with poor label coverage
        if maxCount/numel(segmentLabels) < minimumLabelCoverage || ~ismember(dominantLabel, [1 2 4 5])
            continue;
        end

        % Extract magnitude-based features
        featureVector = computeMagnitudeFeatures(segmentX, segmentY, segmentZ, targetSamplingRateHz);
        WindowedFeatures = [WindowedFeatures; featureVector];
        WindowedLabels = [WindowedLabels; dominantLabel];
        WindowedTimes = [WindowedTimes; sampleTimes(startIdx)];
    end

    % Add contextual (long-term) rolling mean features
    contextWindows = (contextWindowMinutes * 60) / windowDurationSeconds;
    contextualMeans = movmean(WindowedFeatures, [contextWindows - 1 0], 'omitnan');
    WindowedFeatures = [WindowedFeatures contextualMeans];
    WindowedFeatures(~isfinite(WindowedFeatures)) = 0;

    % Store processed data
    ParticipantData.FeatureMatrix = WindowedFeatures;
    ParticipantData.LabelVector = WindowedLabels;
    ParticipantData.TimeVector = WindowedTimes;
catch ME
    warning('Participant %d failed: %s', subjectIndex, ME.message);
end
end

function datetimeVector = parseTimeFlexible(timeColumn)
% Converts various timestamp formats to MATLAB datetime format
    if isdatetime(timeColumn)
        datetimeVector = timeColumn; return;
    end
    if iscellstr(timeColumn) || isstring(timeColumn)
        strTimes = string(timeColumn);
        formats = ["dd-MMM-yyyy HH:mm:ss", "dd/MM/yyyy HH:mm:ss", ...
                   "yyyy-MM-dd HH:mm:ss", "MM/dd/yyyy hh:mm:ss a"];
        for fmt = formats
            try
                datetimeVector = datetime(strTimes, 'InputFormat', fmt, 'TimeZone', 'local'); 
                return;
            end
        end
        datetimeVector = datetime(strTimes, 'TimeZone', 'local');
    elseif isnumeric(timeColumn)
        try
            datetimeVector = datetime(timeColumn, 'ConvertFrom', 'excel', 'TimeZone', 'local');
        catch
            datetimeVector = datetime(timeColumn, 'ConvertFrom', 'posixtime', 'TimeZone', 'local');
        end
    else
        error('Unsupported time column format');
    end
end

function featureVector = computeMagnitudeFeatures(accelX, accelY, accelZ, samplingRateHz)
% Computes 10 orientation-invariant features from acceleration magnitude:
%   1. Mean magnitude
%   2. Standard deviation
%   3. Range
%   4. RMS jerk
%   5. Mean crossing rate
%   6. Dominant frequency
%   7. Bandpower ratio
%   8. Peak power
%   9. Peak area
%  10. Peak slope

    AccelerationMagnitude = sqrt(accelX(:).^2 + accelY(:).^2 + accelZ(:).^2);
    AccelerationMagnitude = AccelerationMagnitude - mean(AccelerationMagnitude);
    sampleCount = numel(AccelerationMagnitude);

    meanMagnitude = mean(abs(AccelerationMagnitude));
    stdMagnitude = std(AccelerationMagnitude);
    rangeMagnitude = max(AccelerationMagnitude) - min(AccelerationMagnitude);
    jerkSignal = diff(AccelerationMagnitude);
    rmsJerk = sqrt(mean(jerkSignal.^2));
    meanCrossRate = sum(diff(sign(AccelerationMagnitude)) ~= 0) / max(sampleCount, 1);

    if sampleCount < 3 || all(AccelerationMagnitude == 0)
        dominantFrequency = 0; bandPowerRatio = 0; peakPower = 0; peakArea = 0; peakSlope = 0;
    else
        powerSpectrum = abs(fft(AccelerationMagnitude)).^2;
        halfIndex = floor(sampleCount / 2) + 1;
        spectrum = powerSpectrum(1:halfIndex);
        frequencyVector = (0:halfIndex-1) / sampleCount * samplingRateHz;
        [~, maxIndex] = max(spectrum);
        cutIndex = max(2, floor(0.2 / samplingRateHz * sampleCount));
        cutIndex = min(cutIndex, halfIndex - 1);
        lowPower = sum(spectrum(2:cutIndex));
        highPower = sum(spectrum(cutIndex+1:end));
        bandPowerRatio = (highPower + eps) / (lowPower + eps);
        leftIndex = max(1, maxIndex - 1);
        rightIndex = min(halfIndex, maxIndex + 1);
        peakPower = spectrum(maxIndex);
        peakArea = sum(spectrum(leftIndex:rightIndex));
        peakSlope = (spectrum(maxIndex) - spectrum(max(1, maxIndex - 1))) / ...
            (frequencyVector(maxIndex) - frequencyVector(max(1, maxIndex - 1)) + eps);
        dominantFrequency = frequencyVector(maxIndex);
    end

    featureVector = [meanMagnitude, stdMagnitude, rangeMagnitude, rmsJerk, meanCrossRate, ...
                     dominantFrequency, bandPowerRatio, peakPower, peakArea, peakSlope];
end