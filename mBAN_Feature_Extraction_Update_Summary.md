# mBAN Feature Extraction - Updated for New Filename Format

## 🎉 **Implementation Complete!**

The mBAN feature extraction system has been successfully updated to handle your new filename formats while maintaining backward compatibility.

## 📝 **What's New**

### **Updated Filename Support**
The system now correctly handles all these filename formats:

#### ✅ **New Formats (with segments)**
- `P001_cabinets2_file03_moving_objects_GlobalSegment1.npy`
- `P001_sitting3_file01_sitting_GlobalSegment1_LocalSegment2.npy`
- `P002_standing1_file01_still_LocalSegment1.npy`
- `P004_stairs_file01_up_GlobalSegment2.npy`

#### ✅ **Original Formats (still supported)**
- `P001_cabinets_file01_drink_coffee.npy`
- `P002_sitting_file01_sit.npy`

#### 🚫 **Failed Files (automatically filtered)**
- `P003_sitting_file01_sitting_GlobalSegment2_LocalSegment1_failed.npy`

## 🔧 **Key Improvements**

### **1. Flexible Activity Mapping**
Added `MBAN_SUB_ACTIVITY_MAP` in constants.py to handle various sub-activity names:

```python
MBAN_SUB_ACTIVITY_MAP = {
    # Sitting variations
    'sitting': 'sit',
    'sit': 'sit',
    
    # Standing variations  
    'standing': 'still',
    'still': 'still',
    'talk': 'talk',
    
    # Cabinet variations
    'coffee': 'coffee',
    'folders': 'folders',
    'objects': 'folders',  # Maps 'moving_objects' to 'folders'
    'moving': 'folders',
    
    # Walking variations
    'walking': 'slow',
    'slow': 'slow',
    'medium': 'medium',
    'fast': 'fast',
    
    # Stairs variations
    'up': 'up',
    'down': 'down',
    'stairs': 'up',
}
```

### **2. Smart Filename Parsing**
The `_get_mban_labels()` function now:
- ✅ Removes segment information (`GlobalSegment1`, `LocalSegment2`)
- ✅ Removes file information (`file01`, `file03`)
- ✅ Strips numbers from activities (`cabinets2` → `cabinets`)
- ✅ Maps sub-activities flexibly (`moving_objects` → `objects` → `folders`)
- ✅ Provides detailed debug output

### **3. Failed File Filtering**
Files ending with `_failed.npy` are automatically excluded from processing.

## 📊 **Label Mapping Results**

| Input Filename | Main Activity | Sub Activity | Main Class | Sub Class |
|----------------|---------------|--------------|------------|-----------|
| `P001_cabinets2_file03_moving_objects_GlobalSegment1.npy` | `cabinets` | `folders` | 1 | 6 |
| `P001_sitting3_file01_sitting_GlobalSegment1_LocalSegment2.npy` | `sitting` | `sit` | 0 | 0 |
| `P002_standing1_file01_still_LocalSegment1.npy` | `standing` | `still` | 1 | 3 |
| `P004_stairs_file01_up_GlobalSegment2.npy` | `stairs` | `up` | 2 | 10 |

## 🚀 **Ready to Use**

Your mBAN feature extraction system is now ready to process all your segmented data formats!

### **Usage Example**
```python
from feature_extractor import extract_mban_features

extract_mban_features(
    data_path="/path/to/your/segmented_mban_data",
    features_data_path="/path/to/output",
    fs=1000,
    window_size=0.5,
    overlap=0.5
)
```

### **Expected Behavior**
- ✅ Processes all good files (with or without segments)
- ✅ Skips all `_failed.npy` files
- ✅ Extracts 45 TSFEL features per 0.5s window
- ✅ Assigns correct activity labels based on filename
- ✅ Generates subject-level feature files + metadata

## 🧪 **Testing Verified**
- ✅ Filename parsing for all formats
- ✅ Failed file filtering
- ✅ Activity-based file filtering  
- ✅ Full feature extraction pipeline
- ✅ Label assignment accuracy

The system is now fully adapted to your new data structure and ready for production use! 🎯
