// Set the side panel as the default action
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });

// Handle extension icon click
chrome.action.onClicked.addListener((tab) => {
  // Open the side panel
  chrome.sidePanel.open({ windowId: tab.windowId });
});

// Log when the extension is installed or updated
chrome.runtime.onInstalled.addListener(() => {
  console.log('Deepfake Image Analyzer extension installed/updated');
}); 