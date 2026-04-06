<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type { Fields, PipelineInfo } from '$lib/types';
  import { PipelineMode } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import ImageInput from '$lib/components/ImageInput.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import Spinner from '$lib/icons/spinner.svelte';
  import Warning from '$lib/components/Warning.svelte';
  import Logging from '$lib/components/Logging.svelte';
  import { lcmLiveStatus, lcmLiveActions, LCMLiveStatus } from '$lib/lcmLive';
  import { mediaStreamActions, onFrameChangeStore, referenceImageStore, referenceImageSent } from '$lib/mediaStream';
  import { getPipelineValues, deboucedPipelineValues } from '$lib/store';
  
  import { FramePusher } from '$lib/framePusher';

  let pipelineParams: Fields;
  let pipelineInfo: PipelineInfo;
  let pageContent: string;
  let isImageMode: boolean = false;
  let maxQueueSize: number = 0;
  let currentQueueSize: number = 0;
  let queueCheckerRunning: boolean = false;
  let warningMessage: string = '';
  let loggingMessage: string = '';

  let engineReady: boolean = false;
  let engineStatus: string = 'Connecting...';
  let statusPollTimer: ReturnType<typeof setTimeout> | null = null;

  let fps: number = 15;
  const framePusher = new FramePusher(lcmLiveActions.send);
  $: framePusher.setFPS(fps);

  let selectedPreset = '';
  const presetImages = [
    '/presets/1.jpeg', 
    '/presets/2.jpeg',
    '/presets/3.jpeg',
    '/presets/4.jpeg',
    '/presets/5.jpeg',
    '/presets/6.jpeg',
    '/presets/7.jpeg',
  ];

  let imageInputComponent: ImageInput;

  onMount(() => {
    pollEngineStatus();
  });

  onDestroy(() => {
    framePusher.stop();
    if (statusPollTimer) clearTimeout(statusPollTimer);
  });

  async function pollEngineStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      engineReady = data.is_ready;
      engineStatus = data.status;
    } catch {
      engineStatus = 'Connecting...';
    }

    if (engineReady) {
      getSettings();
    } else {
      statusPollTimer = setTimeout(pollEngineStatus, 1000);
    }
  }

  async function getSettings() {
    const settings = await fetch(`/api/settings`).then((r) => r.json());
    pipelineParams = settings.input_params.properties;
    pipelineInfo = settings.info.properties;
    isImageMode = pipelineInfo.input_mode.default === PipelineMode.IMAGE;
    maxQueueSize = settings.max_queue_size;
    pageContent = settings.page_content;
    toggleQueueChecker(true);
  }
  function toggleQueueChecker(start: boolean) {
    queueCheckerRunning = start && maxQueueSize > 0;
    if (start) {
      getQueueSize();
    }
  }
  async function getQueueSize() {
    if (!queueCheckerRunning) {
      return;
    }
    const data = await fetch(`/api/queue`).then((r) => r.json());
    currentQueueSize = data.queue_size;
    setTimeout(getQueueSize, 10000);
  }

  function getSreamdata() {
    if (isImageMode) {
      return [$onFrameChangeStore?.blob];
    } else {
      return [];
    }
  }

  async function selectPreset(imageUrl: string) {
    selectedPreset = imageUrl;
    try {
      const res = await fetch(imageUrl);
      const blob = await res.blob();
      if (imageInputComponent) {
        imageInputComponent.loadBlob(blob);
      }
    } catch (e) {
      console.error("Error loading preset:", e);
    }
  }

  let disable_reset = true;
  async function reset() {
    if (imageInputComponent) imageInputComponent.clear();
    selectedPreset = ''; 

    try {
        const response = await fetch('/api/reset', {
            method: 'POST',
        });

        if (response.ok) {
            loggingMessage = 'Successfully reset.';
            disable_reset = true;
        } else {
            warningMessage = "Reset failed.";
        }
    } catch (e) {
        console.error("Network error:", e);
    }
  }

  async function sendReference() {
    const refBlob = $referenceImageStore?.blob;
    if (!refBlob || refBlob.size === 0) {
      warningMessage = 'Please select reference portrait first.';
      return;
    }

    const form = new FormData();
    form.append("ref_image", refBlob, "reference.jpg");

    const res = await fetch(`/api/upload_reference_image`, {
      method: "POST",
      body: form
    });

    if (!res.ok) {
      if (res.status === 503) {
        warningMessage = "Engine is still warming up. Please wait.";
      } else {
        warningMessage = "Failed to upload reference image.";
      }
      return;
    }
    referenceImageSent.set(true);
    loggingMessage = 'Successfully uploaded reference portrait.';
    warningMessage = "";
    disable_reset = false;
  }

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED && $lcmLiveStatus !== LCMLiveStatus.PAUSED;
  $: if ($lcmLiveStatus === LCMLiveStatus.TIMEOUT) {
    warningMessage = 'Session timed out. Please try again.';
    framePusher.stop();
  }

  let disabled = false;
  async function toggleLcmLive() {
    try {
      const refFlag = $referenceImageSent;
      if (!refFlag) {
        warningMessage = 'Please fuse reference portrait first.';
        return;
      }

      if (!isLCMRunning) {
        if (isImageMode) {
          await mediaStreamActions.enumerateDevices();
          await mediaStreamActions.start();

          await new Promise(r => setTimeout(r, 1000));
        }
        disabled = true;
        
        await lcmLiveActions.start(); 
        
        framePusher.start();

        disabled = false;
        toggleQueueChecker(false);
      } else {
        if (isImageMode) {
          mediaStreamActions.stop();
        }
        disabled = true;
        
        framePusher.stop();
        lcmLiveActions.pause();
        
        disabled = false;
        toggleQueueChecker(true);
      }
    } catch (e) {
      warningMessage = e instanceof Error ? e.message : '';
      disabled = false;
      framePusher.stop();
      toggleQueueChecker(true);
    }
  }

  const STATUS_STAGES: Record<string, { icon: string; progress: number }> = {
    'Initializing...':          { icon: '⚙️', progress: 5 },
    'Connecting...':            { icon: '🔌', progress: 0 },
    'Loading models...':        { icon: '🧠', progress: 30 },
    'Warming up DirectML...':   { icon: '🔥', progress: 80 },
    'Ready':                    { icon: '✅', progress: 100 },
  };

  $: currentStage = STATUS_STAGES[engineStatus] ?? { icon: '⏳', progress: 50 };
</script>

<svelte:head>
  <script
    src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js"
  ></script>
</svelte:head>

{#if !engineReady}
  <main class="engine-loading">
    <div class="loading-card">
      <div class="loading-logo">🎭</div>
      <h1 class="loading-title">PersonaLive</h1>
      <p class="loading-subtitle">AMD DirectML Edition</p>

      <div class="loading-stage">
        <span class="stage-icon">{currentStage.icon}</span>
        <span class="stage-text">{engineStatus}</span>
      </div>

      <div class="progress-track">
        <div class="progress-fill" style="width: {currentStage.progress}%"></div>
      </div>

      <p class="loading-hint">Model weights are being loaded into VRAM...</p>
    </div>
  </main>
{:else}
  <main class="container mx-auto flex max-w-5xl flex-col gap-3 px-4 py-4">
    <article class="text-center">
      {#if pageContent}
        {@html pageContent}
      {/if}
      {#if maxQueueSize > 0}
        <p class="text-sm">
          There are <span id="queue_size" class="font-bold">{currentQueueSize}</span>
          user(s) sharing the same GPU, affecting real-time performance. Maximum queue size is {maxQueueSize}.
          <a
            href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
            target="_blank"
            class="text-blue-500 underline hover:no-underline">Duplicate</a
          > and run it on your own GPU.
        </p>
      {/if}
    </article>
    {#if pipelineParams}
      <article class="my-3 grid grid-cols-1 sm:grid-cols-[325px_1fr] gap-8">
        <div class="flex flex-col gap-2">
          <div class="mt-1">
            <div class="grid grid-cols-7 sm:grid-cols-7 gap-1">
              {#each presetImages as imgUrl}
                <button 
                  on:click={() => selectPreset(imgUrl)}
                  class="relative aspect-square w-full rounded-md overflow-hidden border-2 transition-all duration-200 group focus:outline-none
                  {selectedPreset === imgUrl ? 'border-blue-500 ring-2 ring-blue-500/50' : 'border-transparent hover:border-gray-300 dark:hover:border-gray-600'}"
                >
                  <img 
                    src={imgUrl} 
                    alt="Preset" 
                    class="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                  />
                  
                  {#if selectedPreset === imgUrl}
                    <div class="absolute inset-0 bg-black/30 flex items-center justify-center animate-in fade-in duration-200">
                      <svg class="w-6 h-6 text-white drop-shadow-md" fill="none" stroke="currentColor" stroke-width="3" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"></path></svg>
                    </div>
                  {/if}
                </button>
              {/each}
            </div>
          </div>
          <div class="aspect-square w-full">
            <ImageInput 
              bind:this={imageInputComponent}
              width={Number(pipelineParams.width.default)}
              height={Number(pipelineParams.height.default)}
            />
          </div>
          {#if isImageMode}
            <div class="aspect-square w-full">
              <VideoInput
                width={Number(pipelineParams.width.default)}
                height={Number(pipelineParams.height.default)}
              />
            </div>
          {/if}
        </div>

        <div class="flex flex-col gap-2">
          <div class="w-full aspect-square">
            <ImagePlayer />
          </div>
          <div class="sm:col-span-2 border border-slate-300 rounded-lg p-4 flex gap-4 justify-center">
            <div class="flex items-center gap-4 flex-1">
                <span class="text-sm font-bold whitespace-nowrap">Driving FPS: {fps}</span>

                <input 
                    type="range" 
                    min="1" 
                    max="30" 
                    step="1" 
                    bind:value={fps} 
                    class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                />
            </div>
            <Button on:click={sendReference} disabled={$referenceImageSent} classList="px-2 py-2 text-sm"> ② Fuse</Button>

            <Button on:click={toggleLcmLive} {disabled} classList="px-2 py-2 text-sm">
              {#if isLCMRunning}
                ③ Stop
              {:else}
                ③ Start
              {/if}
            </Button>

            <Button on:click={reset} disabled={isLCMRunning||disable_reset} classList="px-2 py-2 text-sm"> Reset </Button>

          </div>
        </div>
      </article>
    {:else}
      <div class="flex items-center justify-center gap-3 py-48 text-2xl">
        <Spinner classList={'animate-spin opacity-50'}></Spinner>
        <p>Loading settings...</p>
      </div>
    {/if}
    <Warning bind:message={warningMessage}></Warning>
    <Logging bind:message={loggingMessage}></Logging>
  </main>
{/if}

<style lang="postcss">
  :global(html) {
    @apply text-black dark:bg-gray-900 dark:text-white;
  }

  .engine-loading {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    overflow: hidden;
  }

  .loading-card {
    text-align: center;
    padding: 3rem 2.5rem;
    border-radius: 1.5rem;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
    max-width: 420px;
    width: 90vw;
    box-shadow: 0 25px 60px rgba(0, 0, 0, 0.5);
    animation: cardFloat 3s ease-in-out infinite;
  }

  @keyframes cardFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
  }

  .loading-logo {
    font-size: 4rem;
    margin-bottom: 0.5rem;
    animation: logoPulse 2s ease-in-out infinite;
  }

  @keyframes logoPulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.1); opacity: 0.85; }
  }

  .loading-title {
    font-size: 2rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.03em;
    margin: 0;
  }

  .loading-subtitle {
    font-size: 0.85rem;
    color: #e74c3c;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.25rem;
    margin-bottom: 2rem;
  }

  .loading-stage {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    min-height: 2rem;
  }

  .stage-icon {
    font-size: 1.3rem;
  }

  .stage-text {
    font-size: 1rem;
    color: rgba(255, 255, 255, 0.8);
    font-weight: 500;
  }

  .progress-track {
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 1.5rem;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #e74c3c, #ff6b6b);
    border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 12px rgba(231, 76, 60, 0.5);
  }

  .loading-hint {
    font-size: 0.75rem;
    color: rgba(255, 255, 255, 0.35);
  }
</style>
