import 'package:flutter/foundation.dart';

class UserData extends ChangeNotifier {
  int? _userId;
  bool _isJudge = true;
  int? _caseId;
  int? get userId => _userId;
  int? get caseId => _caseId;
  
  set userId(int? value) {
    _userId = value;
    notifyListeners(); // Notify listeners of the change
  }

  set caseId(int? value) {
    _caseId = value;
    notifyListeners();
  }

  bool get isJudge => _isJudge;
  set isJudge(bool value) {
    _isJudge = value;
    notifyListeners();
  }
}
