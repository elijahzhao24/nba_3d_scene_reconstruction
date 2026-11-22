function demoClip(id, title, description, directory, videoFile, povTracks) {
  const url = `${import.meta.env.BASE_URL}demos/${directory}/`;
  return {
    id,
    title,
    description,
    sceneUrl: `${url}scene.json`,
    ballUrl: `${url}ball.json`,
    videoUrl: `${url}${videoFile}`,
    povTracks,
  };
}

export const demoClips = [
  demoClip('clip-1', 'Demo 1', 'Luka Dončić step-back game-winner', 'test_video', 'source.webm', [6, 2]),
  demoClip('clip-2', 'Demo 2', 'LeBron James 3-point game-winner', 'test_video_2', 'source.mp4', [5, 9]),
];
