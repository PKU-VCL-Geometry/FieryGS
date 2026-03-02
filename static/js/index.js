$(document).ready(function() {
  // TabsWidget class to control tab switching
  class TabsWidget {
      constructor(container) {
          this.container = container;
          this.activeIndex = 0;
          this.listItems = container.children('.tabs').children('ul').children('li');
          let self = this;

          // Add click event to each tab
          this.listItems.click(function (e) {
              let index = $(this).index();
              self.update($(this), index);
          });

          // Initialize with the first tab active
          this.update(this.listItems[this.activeIndex], this.activeIndex);
      }

      update(element, targetIndex) {
          this.activeIndex = targetIndex;
          const tabs = this.container.children('.tabs');
          const tabsContent = this.container.children('.tabs-content');

          // Update active tab
          this.listItems.each(function () {
              if ($(this).index() == targetIndex) {
                  $(this).addClass('is-active');
              } else {
                  $(this).removeClass('is-active');
              }
          });

          // Show the corresponding video and hide others
          tabsContent.children().each(function () {
              if ($(this).index() == targetIndex) {
                  $(this).show();
                  $(this).find('*').each(function () {
                      if ($(this).is(':visible')) {
                          $(this).trigger('tab:show');
                      }
                  })
              } else {
                  $(this).hide();
                  $(this).find('*').trigger('tab:hide');
              }
          });

          // Dynamically update video sources based on active tab
          updateVideos(targetIndex);
      }
  }

  // Initialize the TabsWidget for the tabs
  $('.tabs-widget').each(function () {
      const containerElement = $(this);
      new TabsWidget(containerElement);
  });

  // Function to update video sources based on tab index
  function updateVideos(tabIndex) {
      const videoSources = [
          ['./static/videos/kitchen.mp4', './static/videos/stool.mp4', './static/videos/playground.mp4', './static/videos/chair.mp4'],
          // ['./static/videos/spot.bridge.mp4', './static/videos/spot.city.mp4', './static/videos/spot.fireplace.mp4', './static/videos/spot.forest.mp4'],
          // ['./static/videos/materials.bridge.mp4', './static/videos/materials.city.mp4', './static/videos/materials.fireplace.mp4', './static/videos/materials.forest.mp4']
      ];

      const videoElements = document.querySelectorAll('.tabs-content .video-container-baseline video');

      videoElements.forEach((video, index) => {
          video.src = videoSources[tabIndex][index];  // Update the video source based on the selected tab
          video.load();  // Reload the video to apply the new source
      });
  }

  // Optionally, if you want video to play/pause based on visibility:
  playPauseVideo();
});

function playPauseVideo() {
let videos = document.querySelectorAll("video");
videos.forEach((video) => {
    // We can only control playback without interaction if the video is muted
    video.muted = true;
    // Play is a promise, so we need to check we have it
    let playPromise = video.play();
    if (playPromise !== undefined) {
        playPromise.then((_) => {
            let observer = new IntersectionObserver(
                (entries) => {
                    entries.forEach((entry) => {
                        if (
                            entry.intersectionRatio !== 1 &&
                            !video.paused
                        ) {
                            video.pause();
                        } else if (video.paused) {
                            video.play();
                        }
                    });
                },
                { threshold: 0.5 }
            );
            observer.observe(video);
        });
    }
});
}
