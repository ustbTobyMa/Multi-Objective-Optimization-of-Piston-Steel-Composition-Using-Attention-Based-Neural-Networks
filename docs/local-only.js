/* Pure local demo restored at the user's request. No model or gateway calls. */
'use strict';
try {
  localStorage.removeItem('piston-gateway-url');
} catch (_) {
  // Local files and private browser modes may disable storage.
}
